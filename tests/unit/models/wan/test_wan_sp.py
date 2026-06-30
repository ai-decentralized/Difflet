"""Megatron-SP wiring + equivalence for difflet.models.wan.modeling_wan.

On the CPU backend every sequence collective is identity (tp==1), so an SP-on
model with the same weights as an SP-off model must produce bit-identical output.
That equivalence is the host-side proof that the SP collectives are placed with
the right logic (gather before column-parallel, reduce-scatter after row-parallel,
sequence scatter/gather at the model boundary). Real multi-rank numerics are
covered by the device parity smoke script.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

_ORIG_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"

import difflet.ops as _ops  # noqa: E402

importlib.reload(_ops)
import difflet.models.wan.modeling_wan as wan  # noqa: E402

importlib.reload(wan)

if _ORIG_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _ORIG_BACKEND


@pytest.fixture(autouse=True)
def _force_cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("DIFFLET_BACKEND", None)
        else:
            os.environ["DIFFLET_BACKEND"] = prev


def _tiny_config(**overrides):
    base = dict(
        patch_size=(1, 2, 2),
        num_attention_heads=4,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=24,
        freq_dim=32,
        ffn_dim=48,
        num_layers=2,
    )
    sp_enabled = overrides.pop("sp_enabled", False)
    base.update(overrides)
    cfg = wan.WanTransformerConfig(**base)
    if sp_enabled:
        cfg.sp_enabled = True
    return cfg


# ------------------------------------------------------------------ threading

def test_attention_threads_sp_flag():
    attn = wan.WanAttention(dim=16, heads=4, head_dim=4, sp_enabled=True)
    assert attn.sp_enabled is True


def test_feed_forward_threads_sp_flag():
    ff = wan.WanFeedForward(dim=16, inner_dim=32, sp_enabled=True)
    assert ff.sp_enabled is True


def test_block_threads_sp_to_attn_and_ffn():
    block = wan.WanTransformerBlock(dim=16, ffn_dim=32, num_heads=4, sp_enabled=True)
    assert block.attn1.sp_enabled is True
    assert block.attn2.sp_enabled is True
    assert block.ffn.sp_enabled is True


def test_model_reads_sp_from_config():
    cfg = _tiny_config(sp_enabled=True)
    model = wan.WanTransformer3DModel(cfg)
    assert model.sp_enabled is True
    assert model.blocks[0].attn1.sp_enabled is True


def test_model_default_sp_is_off():
    model = wan.WanTransformer3DModel(_tiny_config())
    assert model.sp_enabled is False


def test_sp_and_cp_mutually_exclusive_at_model():
    cfg = _tiny_config()
    cfg.sp_enabled = True
    cfg.context_parallel_enabled = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        wan.WanTransformer3DModel(cfg)


# ----------------------------------------------------------------- equivalence

def _clone_with_sp(model_off, cfg_on):
    model_on = wan.WanTransformer3DModel(cfg_on)
    model_on.load_state_dict(model_off.state_dict())
    model_on.eval()
    return model_on


def test_sp_attention_matches_dense_self_attention():
    torch.manual_seed(0)
    off = wan.WanAttention(dim=16, heads=4, head_dim=4, sp_enabled=False).eval()
    on = wan.WanAttention(dim=16, heads=4, head_dim=4, sp_enabled=True).eval()
    on.load_state_dict(off.state_dict())

    hidden = torch.randn(1, 6, 16)
    rope = wan.WanRotaryPosEmbed(attention_head_dim=4, patch_size=(1, 1, 1), max_seq_len=16)
    cos, sin = rope(torch.randn(1, 4, 1, 6, 1))
    with torch.no_grad():
        assert torch.allclose(off(hidden, None, (cos, sin)), on(hidden, None, (cos, sin)), atol=1e-6)


def test_sp_cross_attention_matches_dense():
    torch.manual_seed(1)
    off = wan.WanAttention(dim=16, heads=4, head_dim=4, is_cross_attention=True, sp_enabled=False).eval()
    on = wan.WanAttention(dim=16, heads=4, head_dim=4, is_cross_attention=True, sp_enabled=True).eval()
    on.load_state_dict(off.state_dict())
    hidden = torch.randn(1, 6, 16)
    enc = torch.randn(1, 7, 16)
    with torch.no_grad():
        assert torch.allclose(off(hidden, enc, None), on(hidden, enc, None), atol=1e-6)


def test_sp_feed_forward_matches_dense():
    torch.manual_seed(2)
    off = wan.WanFeedForward(dim=16, inner_dim=32, sp_enabled=False).eval()
    on = wan.WanFeedForward(dim=16, inner_dim=32, sp_enabled=True).eval()
    on.load_state_dict(off.state_dict())
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        assert torch.allclose(off(x), on(x), atol=1e-6)


def test_sp_full_model_matches_dense_t2v():
    torch.manual_seed(0)
    off = wan.WanTransformer3DModel(_tiny_config()).eval()
    on = _clone_with_sp(off, _tiny_config(sp_enabled=True))

    hidden = torch.randn(1, 4, 2, 8, 8)
    timestep = torch.tensor([3.0])
    enc = torch.randn(1, 5, 24)
    with torch.no_grad():
        out_off = off(hidden, timestep, enc)
        out_on = on(hidden, timestep, enc)
    assert out_on.shape == out_off.shape
    assert torch.allclose(out_off, out_on, atol=1e-6)


def test_sp_full_model_matches_dense_ti2v_per_token_modulation():
    # 2D timestep exercises the per-token modulation scatter (timestep_proj/temb).
    torch.manual_seed(0)
    off = wan.WanTransformer3DModel(_tiny_config()).eval()
    on = _clone_with_sp(off, _tiny_config(sp_enabled=True))

    hidden = torch.randn(1, 4, 2, 8, 8)
    seq = 2 * 4 * 4
    timestep = torch.randint(0, 1000, (1, seq)).float()
    enc = torch.randn(1, 5, 24)
    with torch.no_grad():
        out_off = off(hidden, timestep, enc)
        out_on = on(hidden, timestep, enc)
    assert torch.allclose(out_off, out_on, atol=1e-6)
