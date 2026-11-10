"""HunyuanVideo 1.5 joint attention and its fused Q/K RMSNorm + RoPE op.

The op is CUDA-only. These tests register a PyTorch reference of the kernel's arithmetic as
its CPU kernel, test that reference and the model wiring on the CPU, and compare the Triton
kernel with the reference on CUDA.
"""

import contextlib
import importlib
import os

import pytest
import torch
from diffusers.models.embeddings import apply_rotary_emb

from tests.unit.backends.test_cuda_hunyuan15_transformer import TINY, _inputs, _pair

joint = modeling = None
_LANE_WIDTH = 8


# The kernel's arithmetic in PyTorch, step by step as the op's docstring states it.


def _fma_fp32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """FP32 ``fma(a, b, c)`` with a single rounding, on any device.

    The FP32 product is exact in FP64 (48 significant bits). The FP64 sum is made
    round-to-odd from its exact TwoSum error, and rounding a round-to-odd value with
    53 >= 24 + 2 bits to FP32 equals rounding the exact ``a * b + c`` once.
    """
    p = a.double() * b.double()
    c = c.double()
    s = p + c
    t = s - p
    err = (p - (s - t)) + (c - t)
    bits = s.view(torch.int64)
    even_and_inexact = (err != 0) & ((bits & 1) == 0) & torch.isfinite(s)
    toward_exact = torch.where((err > 0) == (s > 0), 1, -1)  # one ulp up or down in magnitude
    return torch.where(even_and_inexact, bits + toward_exact, bits).view(torch.float64).float()


def _sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    """Step 1 over the last dim of FP32 ``x``: lanes of eight FMA chains, then the pair tree."""
    lanes = x.unflatten(-1, (x.shape[-1] // _LANE_WIDTH, _LANE_WIDTH))
    acc = torch.zeros_like(lanes[..., 0])
    for e in range(_LANE_WIDTH):
        acc = _fma_fp32(lanes[..., e], lanes[..., e], acc)
    while acc.shape[-1] > 1:
        acc = acc[..., 0::2] + acc[..., 1::2]
    return acc[..., 0]


def _normalize(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Steps 1-3: FP32 values of ``(x * inv) * w`` rounded once to ``x.dtype``."""
    xf = x.float()
    inv = torch.rsqrt(_sum_of_squares(xf) / xf.shape[-1] + eps)
    return ((xf * inv[..., None]) * weight.float()).to(x.dtype).float()


def _rotate_pairs(n: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Step 4 in FP32: ``(a, b) -> (a*c - b*s, b*c + a*s)`` on adjacent pairs."""
    a, b = n[..., 0::2], n[..., 1::2]
    even = a * cos - b * sin
    odd = b * cos + a * sin
    return torch.stack((even, odd), dim=-1).flatten(-2)


def _reference(q0, k0, v0, q1, k1, v1, wq0, wk0, wq1, wk1, table, eps):
    """The op in PyTorch, rounding to the inputs' dtype where the kernel rounds to BF16:
    identical for BF16 inputs; for FP32 inputs, the Diffusers chain to FP32 rounding."""
    joint_shape = (q0.shape[0], q0.shape[1] + q1.shape[1], *q0.shape[2:])
    outputs = tuple(torch.empty(joint_shape, dtype=q0.dtype, device=q0.device) for _ in range(3))
    batch, joint_len, _, head_dim = joint_shape
    rows = table.float().reshape(batch, joint_len, 1, head_dim)
    cos, sin = rows[..., : head_dim // 2], rows[..., head_dim // 2 :]
    for out, x0, x1, w0, w1 in zip(outputs, (q0, k0), (q1, k1), (wq0, wk0), (wq1, wk1)):
        normed = torch.cat((_normalize(x0, w0, eps), _normalize(x1, w1, eps)), dim=1)
        out.copy_(_rotate_pairs(normed, cos, sin))  # one rounding to the output dtype
    s0 = q0.shape[1]
    outputs[2][:, :s0].copy_(v0)
    outputs[2][:, s0:].copy_(v1)
    return outputs


_registered = False


@pytest.fixture(autouse=True)
def _cpu_ops(monkeypatch):
    # Bind difflet.ops to the CPU ops before the modules import it (a host with no
    # backend set may default to Trainium), then import them lazily.
    global joint, modeling, _registered
    monkeypatch.setenv("DIFFLET_BACKEND", os.environ.get("DIFFLET_BACKEND") or "cpu")
    from difflet.ops import init_parallel_mesh
    from difflet.pipeline.parallel_mesh import MeshSpec

    init_parallel_mesh(MeshSpec(tp=1))
    joint = importlib.import_module("difflet.backends.cuda.hunyuan_video_15.qk_norm_rope")
    modeling = importlib.import_module("difflet.backends.cuda.hunyuan_video_15.modeling")
    if not _registered:
        joint.fused_joint_qk_norm_rope.register_kernel("cpu")(_reference)
        _registered = True
    # The joint path pins cuDNN SDPA, which CPU tensors cannot use.
    monkeypatch.setattr(modeling, "sdpa_kernel", lambda backends: contextlib.nullcontext())


def _tables(s, d, dtype=torch.float32, seed=5):
    g = torch.Generator().manual_seed(seed)
    angles = torch.rand(s, d // 2, generator=g, dtype=torch.float64) * 6.28
    cos = angles.cos().to(dtype).repeat_interleave(2, dim=-1)
    sin = angles.sin().to(dtype).repeat_interleave(2, dim=-1)
    return cos, sin


def test_pack_rope_table_is_the_native_layout():
    cos, sin = _tables(6, 8)
    table = joint.pack_rope_table(cos, sin, 3, torch.bfloat16)
    assert table.shape == (9, 8) and table.dtype == torch.bfloat16
    assert torch.equal(table[:6, :4], cos[:, 0::2].bfloat16())
    assert torch.equal(table[:6, 4:], sin[:, 0::2].bfloat16())
    assert torch.equal(table[6:, :4], torch.ones(3, 4, dtype=torch.bfloat16))
    assert torch.equal(table[6:, 4:], torch.zeros(3, 4, dtype=torch.bfloat16))
    assert joint.pack_rope_table(cos, sin, 0, torch.float32).shape == (6, 8)


def test_joint_op_reference_equals_per_stream_norm_cat_and_rope_in_fp32():
    # FP32 table: the op's eager reference is RMSNorm per stream, cat, interleaved RoPE
    # on the video rows and identity on the context rows (the Diffusers chain's math).
    g = torch.Generator().manual_seed(11)
    s0, s1, h, d = 6, 3, 2, 8
    q0, k0, v0 = (torch.randn(1, s0, h, d, generator=g) for _ in range(3))
    q1, k1, v1 = (torch.randn(1, s1, h, d, generator=g) for _ in range(3))
    w = [torch.rand(d, generator=g) + 0.5 for _ in range(4)]
    cos, sin = _tables(s0, d)
    table = joint.pack_rope_table(cos, sin, s1, torch.float32)
    q, k, v = joint.fused_joint_qk_norm_rope(q0, k0, v0, q1, k1, v1, *w, table, 1e-6)
    assert q.shape == k.shape == v.shape == (1, s0 + s1, h, d)
    norm = lambda x, weight: torch.nn.functional.rms_norm(x, (d,), weight, 1e-6)  # noqa: E731
    expected_q = torch.cat(
        (apply_rotary_emb(norm(q0, w[0]), (cos, sin), sequence_dim=1), norm(q1, w[2])), dim=1
    )
    expected_k = torch.cat(
        (apply_rotary_emb(norm(k0, w[1]), (cos, sin), sequence_dim=1), norm(k1, w[3])), dim=1
    )
    torch.testing.assert_close(q, expected_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(k, expected_k, rtol=1e-6, atol=1e-6)
    assert torch.equal(v, torch.cat((v0, v1), dim=1))


def test_joint_op_is_one_opaque_op_under_compile():
    s0, s1, h, d = 6, 3, 2, 8
    streams = [torch.randn(1, s, h, d) for s in (s0, s0, s0, s1, s1, s1)]
    weights = [torch.ones(d) for _ in range(4)]
    cos, sin = _tables(s0, d)
    table = joint.pack_rope_table(cos, sin, s1, torch.float32)

    def call(*args):
        q, k, v = joint.fused_joint_qk_norm_rope(*args, table, 1e-6)
        return q + k + v

    torch._dynamo.reset()
    compiled = torch.compile(call, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*streams, *weights), call(*streams, *weights))
    # The fake (new_empty joint-shaped tensors) agrees with the real op, and the schema
    # declares no mutation or aliasing.
    torch.library.opcheck(
        joint.fused_joint_qk_norm_rope,
        (*streams, *weights, table, 1e-6),
        test_utils=("test_schema", "test_faketensor"),
    )


def _packed(model):
    model.pack_attention_projections()
    return model


def test_packing_keeps_keys_values_and_memory():
    _, candidate = _pair()
    before = {k: v.clone() for k, v in candidate.state_dict().items()}
    _packed(candidate)
    after = candidate.state_dict()
    assert after.keys() == before.keys()  # packed buffers are not persistent
    assert all(torch.equal(after[k], before[k]) for k in before)
    attn = candidate.transformer_blocks[0].attn
    packed = attn._packed_qkv_weight
    assert packed.shape == (3 * attn.to_q.weight.shape[0], attn.to_q.weight.shape[1])
    # The per-projection weights are views of the packed tensor: no second copy.
    base = packed.untyped_storage().data_ptr()
    for name in ("to_q", "to_k", "to_v"):
        assert getattr(attn, name).weight.untyped_storage().data_ptr() == base
    assert attn.add_v_proj.bias.untyped_storage().data_ptr() == (
        attn._packed_added_qkv_bias.untyped_storage().data_ptr()
    )


def _random_norms(*models):
    # Diffusers initialises RMSNorm weights to ones; randomise them so the test sees the
    # four norm weights' wiring (and eps), identically in both models.
    g = torch.Generator().manual_seed(23)
    for blocks in zip(*(m.transformer_blocks for m in models)):
        for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            w = torch.rand(getattr(blocks[0].attn, name).weight.shape, generator=g) + 0.5
            for block in blocks:
                getattr(block.attn, name).weight.data.copy_(w)


@pytest.mark.parametrize("valid", [(5, 4), (2, 0), (3, 2)])
def test_joint_path_equals_the_prefix_path_in_fp32(valid, monkeypatch):
    _, prefix = _pair()
    _, candidate = _pair()
    _random_norms(prefix, candidate)
    assert all(b.attn.norm_q.eps == 1e-6 for b in candidate.transformer_blocks)
    _packed(candidate)  # packed projections take the joint path, here on the CPU
    calls = []
    real = modeling.fused_joint_qk_norm_rope
    monkeypatch.setattr(
        modeling, "fused_joint_qk_norm_rope", lambda *a: calls.append(a[3].shape[1]) or real(*a)
    )
    inputs = _inputs(text_valid=valid[0], glyph_valid=valid[1])
    with torch.no_grad():
        expected = prefix(**inputs, return_dict=False)[0]
        actual = candidate(**inputs, return_dict=False)[0]
    # One call per block, each with exactly the valid context rows (the native trim).
    assert calls == [valid[0] + valid[1]] * TINY["num_layers"]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_joint_path_needs_packed_projections_and_one_sample(monkeypatch):
    _, candidate = _pair()
    calls = []
    monkeypatch.setattr(modeling, "fused_joint_qk_norm_rope", lambda *a: calls.append(1))
    with torch.no_grad():
        candidate(**_inputs(), return_dict=False)  # not packed: prefix path
        _packed(candidate)
        candidate(**_inputs(batch=2), return_dict=False)  # two samples: prefix path
    assert calls == []


@pytest.mark.parametrize("torch_compile", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_load_packs_only_eager_blocks_on_a_supported_device(monkeypatch, torch_compile, supported):
    # The load-time swap hook runs on the device, before compile (GpuComponent). Compiled
    # blocks keep the prefix path: a trimmed context would recompile per prompt length.
    from difflet.backends.cuda.hunyuan_video_15.transformer import (
        CudaHunyuanVideo15TransformerApplication,
    )

    _, candidate = _pair()
    monkeypatch.setattr(modeling, "joint_attention_supported", lambda weight, d: supported)
    app = CudaHunyuanVideo15TransformerApplication.__new__(CudaHunyuanVideo15TransformerApplication)
    app.torch_compile = torch_compile
    app.apply_load_time_swaps(candidate)
    packed = hasattr(candidate.transformer_blocks[0].attn, "_packed_qkv_weight")
    assert packed == (supported and not torch_compile)


def _f32(value):
    """Round a Python float (a double) to FP32, nearest-even."""
    return torch.tensor(value, dtype=torch.float32).item()


def _explicit_sum_of_squares(row):
    # Every double operation is exact for the rows used here, so each _f32 is the single
    # rounding of one lane FMA or one tree add.
    lanes = []
    for j in range(len(row) // 8):
        acc = 0.0
        for x in row[8 * j : 8 * j + 8]:
            acc = _f32(acc + x * x)
        lanes.append(acc)
    while len(lanes) > 1:
        lanes = [_f32(lanes[i] + lanes[i + 1]) for i in range(0, len(lanes), 2)]
    return lanes[0]


def test_sum_of_squares_is_lane_fmas_then_a_pairwise_tree():
    d = 32  # four lanes
    # Lane 0 is 2^24 then seven ties that round back to 2^24; the other lanes are 1 each, so
    # the tree gives (2^24 + 1) + (1 + 1) = 2^24 + 2 (any other order or lane split differs).
    order = [2.0**12] + [1.0] * 7 + ([1.0] + [0.0] * 7) * 3
    # Lane 0: 2^-24, 2^-23, then fma(1 + 2^-12, 1 + 2^-12, 2^-23): one rounding of
    # 1 + 2^-11 + 3 * 2^-24 (a tie, to even). Multiply-then-add would give 1 + 2^-11 + 2^-23.
    fused = [2.0**-12, 2.0**-12, 1 + 2.0**-12] + [0.0] * 29
    g = torch.Generator().manual_seed(17)
    small = (torch.randint(-8, 9, (2, d), generator=g) / 4).tolist()  # exact everywhere
    # Lanes [2^24, 1, 0, 1]: adjacent-first gives (2^24 + 1) + (0 + 1) = 2^24, while a
    # halves-first (shuffle-down) pairing gives (2^24 + 0) + (1 + 1) = 2^24 + 2.
    halves = [2.0**12] + [1.0] * 7 + [1.0] + [0.0] * 7 + [0.0] * 8 + [1.0] + [0.0] * 7
    rows = [order, fused, halves, *small]
    expected = [_explicit_sum_of_squares(row) for row in rows]
    assert expected[:3] == [2.0**24 + 2, 1 + 2.0**-11 + 2.0**-22, 2.0**24]
    assert _f32(sum(x * x for x in order)) == 2.0**24 + 10  # the exact sum, for contrast

    x = torch.tensor(rows)[None, :, None, :]  # [B=1, S, H=1, D]
    assert _sum_of_squares(x)[0, :, 0].tolist() == expected
    # Through the op: context rows are identity rows, so with FP32 inputs the output is
    # (x * rsqrt(ss / D + eps)) * w with exactly this ss, multiplied in that order.
    video = torch.randn(1, 2, 1, d, generator=g)
    table = joint.pack_rope_table(*_tables(2, d), len(rows), torch.float32)
    wq1, wk1 = (torch.rand(d, generator=g) * 3 + 0.25 for _ in range(2))
    weights = [torch.ones(d), torch.ones(d), wq1, wk1]
    q, k, _ = joint.fused_joint_qk_norm_rope(video, video, video, x, x, x, *weights, table, 1e-6)
    inv = torch.rsqrt(torch.tensor(expected) / d + 1e-6)
    scaled = x * inv[None, :, None, None]
    assert torch.equal(q[:, 2:], scaled * wq1) and torch.equal(k[:, 2:], scaled * wk1)
    assert not torch.equal(q[:, 2:], x * (inv[None, :, None, None] * wq1))  # order is pinned


def test_fma_is_emulated_with_one_rounding():
    t = lambda *v: torch.tensor(v, dtype=torch.float32)  # noqa: E731
    a, c = t(1 + 2.0**-12), t(-(1 + 2.0**-11))
    assert _fma_fp32(a, a, c).item() == 2.0**-24
    assert (a * a + c).item() == 0.0  # two roundings lose it
    # a * b = 1 + 2^-33 sits just above the FP32 midpoint 2^24 + 1; plain FP64 rounds onto
    # the midpoint and then to even (2^24), the single rounding goes up.
    a, b, c = t(1 + 2.0**-11), t(1 - 2.0**-11 + 2.0**-22), t(2.0**24)
    assert (a.double() * b.double() + c.double()).float().item() == 2.0**24
    assert _fma_fp32(a, b, c).item() == 2.0**24 + 2
    special = _fma_fp32(t(float("inf"), float("nan")), t(1.0, 1.0), t(0.0, 0.0))
    assert special[0].item() == float("inf") and special[1].isnan()


def _rotate(n, cos, sin):
    a, b = n[..., 0::2], n[..., 1::2]
    return torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).flatten(-2)


def test_one_bf16_rounding_before_the_rotation():
    g = torch.Generator().manual_seed(29)
    s0, s1, h, d = 4, 2, 2, 8

    def signs(s):  # +-1: ss = D exactly, so with eps = 0 the inverse RMS is exactly 1
        return (torch.randint(0, 2, (1, s, h, d), generator=g) * 2 - 1).bfloat16()

    q0, k0, v0, q1, k1, v1 = (signs(s) for s in (s0, s0, s0, s1, s1, s1))
    w = [torch.rand(d, generator=g) + 0.5 for _ in range(4)]  # FP32: n has bits BF16 drops
    table = joint.pack_rope_table(*_tables(s0, d), s1, torch.float32)
    q, k, _ = joint.fused_joint_qk_norm_rope(q0, k0, v0, q1, k1, v1, *w, table, 0.0)
    cos, sin = table[None, :, None, : d // 2], table[None, :, None, d // 2 :]
    for out, x0, x1, w0, w1 in ((q, q0, q1, w[0], w[2]), (k, k0, k1, w[1], w[3])):
        n = torch.cat((x0.float() * w0, x1.float() * w1), dim=1)
        expected = _rotate(n.bfloat16().float(), cos, sin).bfloat16()
        assert torch.equal(out, expected)
        assert not torch.equal(out, _rotate(n, cos, sin).bfloat16())  # rounding is observable


def test_bf16_rounding_is_nearest_even():
    # x = 1 and eps = 0 make the inverse RMS exactly 1, so n is the weight rounded once to
    # BF16 (ulp 2^-7 on [1, 2)); context rows are identity rows, so the output is n.
    d = 8
    ones = torch.ones(1, 1, 1, d, dtype=torch.bfloat16)
    w = torch.tensor([1 + 2**-8, 1 + 3 * 2**-8, -(1 + 3 * 2**-8), 1 + 2**-8 + 2**-20, 1, 1, 1, 1])
    table = joint.pack_rope_table(*_tables(1, d), 1, torch.float32)
    q, _, _ = joint.fused_joint_qk_norm_rope(
        ones, ones, ones, ones, ones, ones, *[w] * 4, table, 0.0
    )
    assert q[0, 1, 0].float().tolist() == [1.0, 1 + 2**-6, -(1 + 2**-6), 1 + 2**-7, 1, 1, 1, 1]


def _packed_streams(batch, s, h, d, generator, device="cpu"):
    """BF16 q, k, v views split from one packed [B, S, 3*H*D] projection, as on CUDA."""
    packed = (torch.randn(batch, s, 3 * h * d, generator=generator) * 2).to(device, torch.bfloat16)
    return [t.unflatten(-1, (h, d)) for t in packed.split(h * d, dim=-1)]


def test_strided_batched_inputs_identity_rows_and_v_gather():
    g = torch.Generator().manual_seed(31)
    batch, s0, s1, h, d, eps = 2, 5, 3, 3, 16, 1e-6
    streams = (*_packed_streams(batch, s0, h, d, g), *_packed_streams(batch, s1, h, d, g))
    q0, k0, v0, q1, k1, v1 = streams
    assert not q0.is_contiguous() and q0.stride() == k0.stride() == v0.stride()
    w = [(torch.rand(d, generator=g) + 0.5).bfloat16() for _ in range(4)]
    tables = [
        joint.pack_rope_table(*_tables(s0, d, seed=b), s1, torch.bfloat16) for b in range(batch)
    ]
    q, k, v = outputs = joint.fused_joint_qk_norm_rope(*streams, *w, torch.cat(tables), eps)
    for out in outputs:
        assert out.shape == (batch, s0 + s1, h, d) and out.dtype == torch.bfloat16
        assert out.is_contiguous()
    # Reading through the strides equals reading contiguous copies.
    dense = joint.fused_joint_qk_norm_rope(
        *(t.contiguous() for t in streams), *w, torch.cat(tables), eps
    )
    assert all(torch.equal(a, b) for a, b in zip(outputs, dense))
    assert torch.equal(v, torch.cat((v0, v1), dim=1))  # V gathered bit for bit
    # Identity context rows leave the normalised values unrotated; video rows are rotated.
    assert torch.equal(q[:, s0:], _normalize(q1, w[2], eps).bfloat16())
    assert torch.equal(k[:, s0:], _normalize(k1, w[3], eps).bfloat16())
    assert not torch.equal(q[:, :s0], _normalize(q0, w[0], eps).bfloat16())
    # Each sample reads its own table rows.
    for b in range(batch):
        one = joint.fused_joint_qk_norm_rope(*(t[b : b + 1] for t in streams), *w, tables[b], eps)
        assert all(torch.equal(full[b : b + 1], part) for full, part in zip(outputs, one))


def _ulps(x):
    """BF16 bit patterns as integers that count ulps through zero."""
    bits = x.view(torch.int16).int()
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernel needs CUDA")
@pytest.mark.parametrize("batch, s0, s1, heads", [(1, 37, 11, 16), (2, 20, 5, 12)])
def test_cuda_kernel_matches_the_reference(batch, s0, s1, heads):
    pytest.importorskip("triton")
    g = torch.Generator().manual_seed(41)
    d = 128
    streams = (
        *_packed_streams(batch, s0, heads, d, g, "cuda"),
        *_packed_streams(batch, s1, heads, d, g, "cuda"),
    )
    w = [(torch.rand(d, generator=g) + 0.5).to("cuda", torch.bfloat16) for _ in range(4)]
    tables = []
    for b in range(batch):
        cos, sin = (t.cuda() for t in _tables(s0, d, seed=b))
        tables.append(joint.pack_rope_table(cos, sin, s1, torch.bfloat16))
    args = (*streams, *w, torch.cat(tables), 1e-6)
    kernel = joint.fused_joint_qk_norm_rope(*args)
    reference = _reference(*args)
    for out in kernel:
        assert out.shape == (batch, s0 + s1, heads, d) and out.dtype == torch.bfloat16
        assert out.is_contiguous()
    assert torch.equal(kernel[2].view(torch.int16), reference[2].view(torch.int16))
    for got, want in zip(kernel[:2], reference[:2]):
        # Identity (context) rows are the normalised values: bitwise.
        assert torch.equal(got[:, s0:].view(torch.int16), want[:, s0:].view(torch.int16))
        assert (_ulps(got) - _ulps(want)).abs().max().item() <= 1


def test_joint_path_needs_cuda_triton_bf16_head_128_and_cudnn(monkeypatch):
    from types import SimpleNamespace as NS

    cuda = torch.device("cuda", 0)
    bf16 = NS(is_cuda=True, dtype=torch.bfloat16, device=cuda)
    monkeypatch.setattr(modeling, "_cudnn_attention_available", lambda device: True)
    monkeypatch.setattr(modeling, "HAS_TRITON", True)
    assert modeling.joint_attention_supported(bf16, 128)
    cpu = NS(is_cuda=False, dtype=torch.bfloat16, device=torch.device("cpu"))
    assert not modeling.joint_attention_supported(cpu, 128)
    fp32 = NS(is_cuda=True, dtype=torch.float32, device=cuda)
    assert not modeling.joint_attention_supported(fp32, 128)
    assert not modeling.joint_attention_supported(bf16, 64)
    monkeypatch.setattr(modeling, "HAS_TRITON", False)
    assert not modeling.joint_attention_supported(bf16, 128)
    monkeypatch.setattr(modeling, "HAS_TRITON", True)
    monkeypatch.setattr(modeling, "_cudnn_attention_available", lambda device: False)
    assert not modeling.joint_attention_supported(bf16, 128)
