"""Megatron-SP wiring + equivalence for the HunyuanVideo DiT.

On the CPU backend every sequence collective is identity (tp==1), so an SP-on
model with the same weights as an SP-off model must produce bit-identical output.
That equivalence is the host-side proof that the SP collectives are placed with
the right logic (gather before column-parallel, reduce-scatter after row-parallel,
sequence scatter/gather at the model boundary, and only the latent/video stream
kept sequence-sharded through the dual-stream blocks while the text stream stays
full/dense and the single-stream blocks run dense). Real multi-rank numerics are
covered by the
device parity smoke script (scripts/hunyuan_sp_parity_smoke.sh).
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
import difflet.models.hunyuan_video.modeling_hunyuan_video as hv  # noqa: E402

importlib.reload(hv)

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


def _rotary(seq, head_dim):
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    return get_1d_rotary_pos_embed(
        head_dim,
        torch.arange(seq, dtype=torch.float32),
        theta=256.0,
        use_real=True,
    )


def _tiny_config(**overrides):
    base = dict(
        in_channels=2,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        patch_size=2,
        patch_size_t=1,
        guidance_embeds=True,
        text_embed_dim=6,
        pooled_projection_dim=5,
        rope_axes_dim=(2, 2, 0),
    )
    base.update(overrides)
    return hv.HunyuanVideoTransformerConfig(**base)


def _tiny_model(sp_enabled=False):
    torch.manual_seed(0)
    return hv.HunyuanVideoTransformer3DModel(
        in_channels=2,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        patch_size=2,
        patch_size_t=1,
        guidance_embeds=True,
        text_embed_dim=6,
        pooled_projection_dim=5,
        rope_axes_dim=(2, 2, 0),
        sp_enabled=sp_enabled,
    ).eval()


def _tiny_inputs():
    torch.manual_seed(1)
    return dict(
        hidden_states=torch.randn(1, 2, 2, 4, 4),
        timestep=torch.tensor([7], dtype=torch.long),
        encoder_hidden_states=torch.randn(1, 4, 6),
        encoder_attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        pooled_projections=torch.randn(1, 5),
        guidance=torch.tensor([3], dtype=torch.long),
    )


# ------------------------------------------------------------------ threading

def test_attention_threads_sp_flag():
    attn = hv.HunyuanVideoAttention(
        hidden_size=8,
        num_attention_heads=2,
        attention_head_dim=4,
        added_kv_proj_dim=8,
        context_pre_only=False,
        pre_only=False,
        sp_enabled=True,
    )
    assert attn.sp_enabled is True


def test_feed_forward_threads_sp_flag():
    ff = hv.HunyuanVideoFeedForward(8, mult=2.0, sp_enabled=True)
    assert ff.sp_enabled is True


def test_dual_block_threads_sp_to_attn_and_ffns():
    block = hv.HunyuanVideoTransformerBlock(
        num_attention_heads=2, attention_head_dim=4, mlp_ratio=2.0, sp_enabled=True
    )
    assert block.sp_enabled is True
    assert block.attn.sp_enabled is True
    # Latent/video-only SP: the latent FF shards, the text FF stays dense.
    assert block.ff.sp_enabled is True
    assert block.ff_context.sp_enabled is False


def test_single_block_threads_sp_to_attn():
    block = hv.HunyuanVideoSingleTransformerBlock(
        num_attention_heads=2, attention_head_dim=4, mlp_ratio=2.0, sp_enabled=True
    )
    assert block.sp_enabled is True
    assert block.attn.sp_enabled is True


def test_model_reads_sp_from_config():
    cfg = _tiny_config(sp_enabled=True)
    model = hv.HunyuanVideoTransformer3DModel(cfg)
    assert model.sp_enabled is True
    assert model.transformer_blocks[0].attn.sp_enabled is True
    assert model.single_transformer_blocks[0].attn.sp_enabled is True


def test_model_reads_sp_from_kwarg():
    model = _tiny_model(sp_enabled=True)
    assert model.sp_enabled is True


def test_model_default_sp_is_off():
    model = _tiny_model()
    assert model.sp_enabled is False
    assert model.transformer_blocks[0].attn.sp_enabled is False


def test_sp_and_cp_mutually_exclusive_at_model():
    cfg = _tiny_config()
    cfg.sp_enabled = True
    cfg.context_parallel_enabled = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        hv.HunyuanVideoTransformer3DModel(cfg)


# ----------------------------------------------------------------- equivalence

def test_sp_dual_attention_matches_dense():
    torch.manual_seed(0)
    heads, hd = 2, 4
    common = dict(
        hidden_size=heads * hd,
        num_attention_heads=heads,
        attention_head_dim=hd,
        added_kv_proj_dim=heads * hd,
        context_pre_only=False,
        pre_only=False,
    )
    off = hv.HunyuanVideoAttention(**common, sp_enabled=False).eval()
    on = hv.HunyuanVideoAttention(**common, sp_enabled=True).eval()
    on.load_state_dict(off.state_dict())

    hs = torch.randn(2, 4, heads * hd)
    enc = torch.randn(2, 3, heads * hd)
    rot = _rotary(4, hd)
    with torch.no_grad():
        off_out, off_ctx = off(hidden_states=hs, encoder_hidden_states=enc, image_rotary_emb=rot)
        on_out, on_ctx = on(hidden_states=hs, encoder_hidden_states=enc, image_rotary_emb=rot)
    assert torch.allclose(off_out, on_out, atol=1e-6)
    assert torch.allclose(off_ctx, on_ctx, atol=1e-6)


def test_sp_feed_forward_matches_dense():
    torch.manual_seed(2)
    off = hv.HunyuanVideoFeedForward(8, mult=2.0, sp_enabled=False).eval()
    on = hv.HunyuanVideoFeedForward(8, mult=2.0, sp_enabled=True).eval()
    on.load_state_dict(off.state_dict())
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        assert torch.allclose(off(x), on(x), atol=1e-6)


def test_sp_dual_block_matches_dense():
    torch.manual_seed(0)
    heads, hd = 2, 4
    off = hv.HunyuanVideoTransformerBlock(heads, hd, mlp_ratio=2.0, sp_enabled=False).eval()
    on = hv.HunyuanVideoTransformerBlock(heads, hd, mlp_ratio=2.0, sp_enabled=True).eval()
    on.load_state_dict(off.state_dict())

    hs = torch.randn(2, 4, heads * hd)
    enc = torch.randn(2, 3, heads * hd)
    temb = torch.randn(2, heads * hd)
    rot = _rotary(4, hd)
    with torch.no_grad():
        off_out, off_ctx = off(hs, enc, temb, None, rot)
        on_out, on_ctx = on(hs, enc, temb, None, rot)
    assert torch.allclose(off_out, on_out, atol=1e-6)
    assert torch.allclose(off_ctx, on_ctx, atol=1e-6)


def test_sp_single_block_matches_dense():
    torch.manual_seed(0)
    heads, hd = 2, 4
    off = hv.HunyuanVideoSingleTransformerBlock(heads, hd, mlp_ratio=2.0, sp_enabled=False).eval()
    on = hv.HunyuanVideoSingleTransformerBlock(heads, hd, mlp_ratio=2.0, sp_enabled=True).eval()
    on.load_state_dict(off.state_dict())

    hs = torch.randn(2, 4, heads * hd)
    enc = torch.randn(2, 3, heads * hd)
    temb = torch.randn(2, heads * hd)
    rot = _rotary(4, hd)
    with torch.no_grad():
        off_out, off_ctx = off(hs, enc, temb, None, rot)
        on_out, on_ctx = on(hs, enc, temb, None, rot)
    assert torch.allclose(off_out, on_out, atol=1e-6)
    assert torch.allclose(off_ctx, on_ctx, atol=1e-6)


def test_sp_full_model_matches_dense():
    off = _tiny_model(sp_enabled=False)
    on = _tiny_model(sp_enabled=True)
    on.load_state_dict(off.state_dict())
    on.eval()

    inputs = _tiny_inputs()
    with torch.no_grad():
        out_off = off(**inputs, return_dict=False)[0]
        out_on = on(**inputs, return_dict=False)[0]
    assert out_on.shape == out_off.shape
    assert torch.allclose(out_off, out_on, atol=1e-6)
