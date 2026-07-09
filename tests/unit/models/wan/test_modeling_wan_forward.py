"""CPU-backend forward-pass coverage for difflet.models.wan.modeling_wan.

These exercise the real nn.Module forwards (transformer block, attention, FFN,
rotary embeds, time/text embedding, top-level transformer and teacache signal)
on the ``cpu`` op backend with tiny deterministic configs.
"""

from __future__ import annotations

import os
import importlib

import pytest
import torch

# Select the CPU op backend BEFORE binding difflet.ops symbols into modeling_wan.
_ORIG_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"

import difflet.ops as _ops  # noqa: E402

importlib.reload(_ops)
import difflet.models.wan.modeling_wan as wan  # noqa: E402

importlib.reload(wan)

# The cpu op symbols are now bound into the reloaded module; restore the original
# backend env so importing this module doesn't leak cpu to later suite modules.
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
    base.update(overrides)
    return wan.WanTransformerConfig(**base)


# ---------------------------------------------------------------------------
# Config


def test_config_inner_dim_and_from_diffusers_dict():
    cfg = _tiny_config()
    assert cfg.inner_dim == 4 * 16
    raw = {
        "patch_size": [1, 2, 2],
        "num_attention_heads": 4,
        "attention_head_dim": 16,
        "unknown_future_field": 123,
    }
    built = wan.WanTransformerConfig.from_diffusers_dict(raw)
    assert built.patch_size == (1, 2, 2)
    assert built.num_attention_heads == 4


def test_config_rejects_non_default_qk_norm():
    with pytest.raises(NotImplementedError):
        _tiny_config(qk_norm="layer_norm")


def test_config_rejects_i2v_fields():
    with pytest.raises(NotImplementedError, match="I2V"):
        _tiny_config(image_dim=512)
    with pytest.raises(NotImplementedError):
        _tiny_config(added_kv_proj_dim=64)
    with pytest.raises(NotImplementedError):
        _tiny_config(pos_embed_seq_len=10)


def test_safe_tp_size_falls_back_to_one():
    assert wan._safe_tp_size() == 1


# ---------------------------------------------------------------------------
# Component forwards


def test_rotary_pos_embed_shapes():
    rope = wan.WanRotaryPosEmbed(attention_head_dim=16, patch_size=(1, 2, 2), max_seq_len=64)
    hidden = torch.randn(1, 4, 2, 8, 8)
    cos, sin = rope(hidden)
    s = 2 * 4 * 4
    assert cos.shape == (1, s, 1, 16)
    assert sin.shape == (1, s, 1, 16)


def test_time_text_embedding_both_timestep_modes():
    emb = wan.WanTimeTextEmbedding(
        dim=64, time_freq_dim=32, time_proj_dim=64 * 6, text_embed_dim=24
    )
    ehs = torch.randn(1, 5, 24)
    temb, proj, out_ehs = emb(torch.tensor([3.0]), ehs)
    assert temb.shape == (1, 64)
    assert proj.shape == (1, 64 * 6)
    assert out_ehs.shape == (1, 5, 64)

    # timestep_seq_len path (Ti2V): timestep flattened over (B*S_t).
    temb2, proj2, _ = emb(torch.arange(4, dtype=torch.float32), ehs, timestep_seq_len=4)
    assert temb2.shape == (1, 4, 64)
    assert proj2.shape == (1, 4, 64 * 6)


def test_feed_forward_forward():
    ff = wan.WanFeedForward(dim=16, inner_dim=32)
    out = ff(torch.randn(1, 5, 16))
    assert out.shape == (1, 5, 16)


def test_attn_kernel_helper_shape():
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 4, 5, 8)
    v = torch.randn(1, 4, 5, 8)
    out = wan._attn_kernel(q, k, v, head_dim=8)
    assert out.shape == (1, 4, 5, 8)


def test_self_attention_forward_with_rotary():
    torch.manual_seed(0)
    attn = wan.WanAttention(dim=16, heads=4, head_dim=4)
    hidden = torch.randn(1, 5, 16)
    rope = wan.WanRotaryPosEmbed(attention_head_dim=4, patch_size=(1, 1, 1), max_seq_len=16)
    cos, sin = rope(torch.randn(1, 4, 1, 5, 1))  # produces S=5 tokens
    out = attn(hidden, None, (cos, sin))
    assert out.shape == (1, 5, 16)


def test_cross_attention_ignores_rotary():
    attn = wan.WanAttention(dim=16, heads=4, head_dim=4, is_cross_attention=True)
    hidden = torch.randn(1, 5, 16)
    enc = torch.randn(1, 7, 16)
    out = attn(hidden, enc, None)
    assert out.shape == (1, 5, 16)


def test_global_rms_norm_matches_plain_rmsnorm():
    attn = wan.WanAttention(dim=16, heads=4, head_dim=4)
    x = torch.randn(1, 5, 16)
    normed = attn._global_rms_norm(attn.norm_q, x)
    assert normed.shape == x.shape


def test_transformer_block_forward_3d_temb():
    torch.manual_seed(0)
    block = wan.WanTransformerBlock(dim=16, ffn_dim=32, num_heads=4)
    hidden = torch.randn(1, 5, 16)
    enc = torch.randn(1, 7, 16)
    # 3D timestep_proj: (B, 6, dim)
    temb = torch.randn(1, 6, 16)
    rope = wan.WanRotaryPosEmbed(attention_head_dim=4, patch_size=(1, 1, 1), max_seq_len=16)
    cos, sin = rope(torch.randn(1, 5, 1, 1, 1))
    out = block(hidden, enc, temb, (cos, sin))
    assert out.shape == (1, 5, 16)


def test_transformer_block_forward_4d_temb():
    torch.manual_seed(0)
    block = wan.WanTransformerBlock(dim=16, ffn_dim=32, num_heads=4)
    hidden = torch.randn(1, 5, 16)
    enc = torch.randn(1, 7, 16)
    # 4D temb: (B, S, 6, dim)
    temb = torch.randn(1, 5, 6, 16)
    rope = wan.WanRotaryPosEmbed(attention_head_dim=4, patch_size=(1, 1, 1), max_seq_len=16)
    cos, sin = rope(torch.randn(1, 5, 1, 1, 1))
    out = block(hidden, enc, temb, (cos, sin))
    assert out.shape == (1, 5, 16)


# ---------------------------------------------------------------------------
# Top-level transformer


def test_transformer_forward_t2v():
    torch.manual_seed(0)
    model = wan.WanTransformer3DModel(_tiny_config())
    hidden = torch.randn(1, 4, 2, 8, 8)
    timestep = torch.tensor([3.0])
    enc = torch.randn(1, 5, 24)
    out = model(hidden, timestep, enc)
    assert out.shape == (1, 4, 2, 8, 8)


def test_transformer_forward_ti2v_2d_timestep():
    torch.manual_seed(0)
    model = wan.WanTransformer3DModel(_tiny_config())
    hidden = torch.randn(1, 4, 2, 8, 8)
    seq = 2 * 4 * 4
    timestep = torch.randint(0, 1000, (1, seq)).float()
    enc = torch.randn(1, 5, 24)
    out = model(hidden, timestep, enc)
    assert out.shape == (1, 4, 2, 8, 8)


def test_transformer_cfg_and_cp_mutually_exclusive():
    cfg = _tiny_config()
    cfg.cfg_parallel_enabled = True
    cfg.context_parallel_enabled = True
    with pytest.raises(ValueError, match="mutually"):
        wan.WanTransformer3DModel(cfg)


def test_teacache_mod_input_1d_timestep():
    torch.manual_seed(0)
    model = wan.WanTransformer3DModel(_tiny_config())
    hidden = torch.randn(1, 4, 2, 8, 8)
    enc = torch.randn(1, 5, 24)
    sig = model.teacache_mod_input(hidden, torch.tensor([3.0]), enc)
    assert sig.shape == (1, 2 * 4 * 4, 64)


def test_teacache_mod_input_2d_timestep():
    torch.manual_seed(0)
    model = wan.WanTransformer3DModel(_tiny_config())
    hidden = torch.randn(1, 4, 2, 8, 8)
    enc = torch.randn(1, 5, 24)
    seq = 2 * 4 * 4
    timestep = torch.randint(0, 1000, (1, seq)).float()
    sig = model.teacache_mod_input(hidden, timestep, enc)
    assert sig.shape == (1, seq, 64)
