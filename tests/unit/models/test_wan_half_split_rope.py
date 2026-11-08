"""Wan's half-split rotary layout is the interleaved one with each self-attention head's q/k
channels reordered to (0, 2, 4, ..., 1, 3, 5, ...): the index, the tables, the rotation and the
whole model agree with the interleaved law to fp32 summation order, and the reordered weights
with the interleaved rotary do not (the control that the reorder reached every q/k)."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from difflet import ops  # noqa: E402
from difflet.models.wan.modeling_wan import (  # noqa: E402
    WanRotaryPosEmbed,
    apply_rotary_emb_half_split,
    half_split_index,
)
from tests.unit.backends._neuron_toy import tp1_mesh  # noqa: E402
from tests.unit.backends._neuron_wan_toy import (  # noqa: E402
    TINY_FP8_WAN,
    TINY_WAN_TRANSFORMER,
    tiny_wan_inputs,
    tiny_wan_tp1_model,
)


def _tables(head_dim: int, shape: tuple[int, ...]):
    rope = WanRotaryPosEmbed(head_dim, (1, 2, 2), max_seq_len=64)
    x = torch.zeros(shape)
    return rope(x), rope(x, layout="half_split")


def _max_rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got - ref).abs().max() / ref.abs().max()).item()


def test_half_split_index_reorders_within_each_head():
    assert half_split_index(8, 4).tolist() == [0, 2, 1, 3, 4, 6, 5, 7]
    assert half_split_index(8, 4).dtype == torch.int64
    with pytest.raises(ValueError, match="head_dim"):
        half_split_index(6, 4)


@pytest.mark.parametrize("head_dim", [128, 32], ids=["wan-44-42-42", "fp8-toy-12-10-10"])
def test_half_split_tables_are_the_interleaved_tables_reordered(head_dim):
    (cos_il, sin_il), (cos_h, sin_h) = _tables(head_dim, (1, 16, 3, 8, 12))
    perm = half_split_index(head_dim, head_dim)
    sign = torch.cat([-torch.ones(head_dim // 2), torch.ones(head_dim // 2)])
    assert cos_h.shape == sin_h.shape == cos_il.shape == (1, 3 * 4 * 6, 1, head_dim)
    assert cos_h.dtype == cos_il.dtype
    assert torch.equal(cos_h, cos_il[..., perm])
    assert torch.equal(sin_h, sign.to(sin_il.dtype) * sin_il[..., perm])


def test_unknown_rope_layout_is_refused():
    rope = WanRotaryPosEmbed(32, (1, 2, 2), max_seq_len=64)
    with pytest.raises(ValueError, match="half_split"):
        rope(torch.zeros(1, 16, 3, 8, 12), layout="split")


def test_half_split_rotation_is_the_interleaved_rotation_reordered():
    (cos_il, sin_il), (cos_h, sin_h) = _tables(128, (1, 16, 4, 8, 12))
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 96, 4, 128, generator=gen)
    perm = half_split_index(128, 128)
    got = apply_rotary_emb_half_split(x[..., perm], cos_h, sin_h)
    ref = ops.apply_rotary_emb(x, cos_il, sin_il)[..., perm]
    assert got.dtype == x.dtype
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=0)


def _output(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        return model(*inputs)


def _set_layout(model, layout: str) -> None:
    model.rope_layout = layout
    for block in model.blocks:
        block.attn1.rope_layout = layout


def _trained_qk_norms(model) -> None:
    """Per-channel q/k norm weights, as a checkpoint has: the toy initializes them to ones, which
    any reorder leaves as they are, so a reorder that missed them would go unseen."""
    gen = torch.Generator().manual_seed(7)
    for block in model.blocks:
        for norm in (block.attn1.norm_q, block.attn1.norm_k):
            norm.weight.data = 0.5 + torch.rand(norm.weight.shape, generator=gen)


@pytest.mark.parametrize("config", [TINY_WAN_TRANSFORMER, TINY_FP8_WAN], ids=["w1", "fp8"])
def test_reordered_model_matches_the_interleaved_model(config):
    with tp1_mesh():
        model = tiny_wan_tp1_model(config=config)
        _trained_qk_norms(model)
        inputs = tiny_wan_inputs(config=config)
        ref = _output(model, inputs)
        reordered = copy.deepcopy(model)
        loaded = reordered.state_dict()  # shares the tensors the reorder must not edit
        snapshot = {name: tensor.clone() for name, tensor in loaded.items()}
        assert reordered.rope_layout == "interleaved"
        assert reordered.use_half_split_rope_() == config["num_layers"]
        assert reordered.rope_layout == "half_split"
        assert all(b.attn1.rope_layout == "half_split" for b in reordered.blocks)
        assert all(b.attn2.rope_layout == "interleaved" for b in reordered.blocks)
        assert all(torch.equal(loaded[name], snapshot[name]) for name in loaded)
        got = _output(reordered, inputs)
        assert _max_rel(got, ref) < 1e-5

        # A second call finds the model in the half-split layout already: nothing moves.
        assert reordered.use_half_split_rope_() == 0
        assert torch.equal(_output(reordered, inputs), got)

        # The control: the reordered weights under the interleaved rotary are a different model.
        _set_layout(reordered, "interleaved")
        assert _max_rel(_output(reordered, inputs), ref) > 1e-3


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slices_op_falls_back_to_the_backends_rotary(monkeypatch, dtype):
    """A backend without its own stride-2 op (every one but neuron) rotates exactly as before."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    (cos, sin), _ = _tables(128, (1, 16, 2, 6, 8))
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 24, 4, 128, generator=gen).to(dtype)
    got = ops.apply_rotary_emb_slices(x, cos, sin)
    assert got.dtype == dtype
    assert torch.equal(got, ops.apply_rotary_emb(x, cos, sin))


def test_wan_interleaved_self_attention_uses_the_slices_op(monkeypatch):
    """The interleaved self-attention rotates q and k with ``apply_rotary_emb_slices`` (the
    stride-2 law the neuron backend keeps for it), the half-split one never does, and on the CPU
    backend the toy is bit for bit what it was with ``apply_rotary_emb``."""
    from difflet.backends.cpu.ops_impl.embeddings import apply_rotary_emb as cpu_rope
    from difflet.models.wan import modeling_wan

    slices = modeling_wan.apply_rotary_emb_slices
    calls = []

    def spy(hidden_states, cos, sin):
        calls.append(hidden_states.shape)
        return slices(hidden_states, cos, sin)

    with tp1_mesh():
        model = tiny_wan_tp1_model()
        inputs = tiny_wan_inputs()
        monkeypatch.setattr(modeling_wan, "apply_rotary_emb_slices", cpu_rope)
        before = _output(model, inputs)
        monkeypatch.setattr(modeling_wan, "apply_rotary_emb_slices", spy)
        got = _output(model, inputs)
        assert len(calls) == 2 * TINY_WAN_TRANSFORMER["num_layers"]
        assert torch.equal(got, before)

        calls.clear()
        model.use_half_split_rope_()
        _output(model, inputs)
        assert calls == []


def test_the_reorder_moves_an_fp8_twins_block_scales_with_its_rows():
    """An MXFP8 twin's q/k carry an fp8 weight and an E8M0 ``weight_scale`` buffer, one row per
    output channel: both move with the rows (each row here holds its own channel's code), and
    the scales stay a registered buffer, so a later ``.to(device)`` still moves them."""
    with tp1_mesh():
        model = tiny_wan_tp1_model(config=TINY_FP8_WAN)
    rows = TINY_FP8_WAN["num_attention_heads"] * TINY_FP8_WAN["attention_head_dim"]
    codes = torch.arange(rows, dtype=torch.uint8)[:, None]
    for block in model.blocks:
        for linear in (block.attn1.to_q, block.attn1.to_k):
            weight = codes.expand(rows, rows).contiguous().view(torch.float8_e4m3fn)
            linear.weight = torch.nn.Parameter(weight, requires_grad=False)
            linear.register_buffer("weight_scale", codes.expand(rows, rows // 32).contiguous())
    model.use_half_split_rope_()
    perm = half_split_index(rows, TINY_FP8_WAN["attention_head_dim"])
    for block in model.blocks:
        for linear in (block.attn1.to_q, block.attn1.to_k):
            assert linear.weight.dtype == torch.float8_e4m3fn
            assert torch.equal(linear.weight.view(torch.uint8), codes[perm].expand(rows, rows))
            assert torch.equal(dict(linear.named_buffers())["weight_scale"],
                               codes[perm].expand(rows, rows // 32))
