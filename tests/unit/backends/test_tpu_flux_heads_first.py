"""FLUX SP's heads-first attention (backends/tpu/flux/sp.py) equals the TP
processor it replaces (models/flux/tp_sharding.FluxTPAttnProcessor), on the CPU.

Same diffusers ``FluxAttention`` (per-head RMSNorm, added text projections),
same q/k/v, same RoPE, float64; both run the attention through the same SDPA so
only the reordering is compared: joint (text + image) and single blocks, FLUX's
interleaved RoPE and the split-halves form (diffusers' and the per-half
``_rope_split_halves``, which must also be bit-identical on its own).
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("diffusers")
from diffusers.models.transformers.transformer_flux import FluxAttention  # noqa: E402

HEADS, HEAD_DIM, TEXT, IMAGE = 3, 16, 5, 7


def _sdpa(q, k, v, *, scale, causal, **_):
    return F.scaled_dot_product_attention(q[None], k[None], v[None], scale=scale,
                                          is_causal=causal)[0]


def _rope(seq):
    angles = torch.randn(seq, HEAD_DIM, dtype=torch.float64)
    return angles.cos(), angles.sin()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_split_halves_is_bit_identical(dtype):
    from diffusers.models.embeddings import apply_rotary_emb

    sp = importlib.import_module("difflet.backends.tpu.flux.sp")
    torch.manual_seed(0)
    x = torch.randn(2, HEADS, IMAGE, HEAD_DIM).to(dtype)
    cos, sin = (t.float() for t in _rope(IMAGE))
    want = apply_rotary_emb(x, (cos, sin), sequence_dim=2, use_real_unbind_dim=-2)
    torch.testing.assert_close(sp._rope_split_halves(x, cos, sin), want, rtol=0, atol=0)


@pytest.mark.parametrize("joint", [True, False])
@pytest.mark.parametrize("unbind,fused", [(-1, False), (-2, False), (-2, True)])
def test_heads_first_matches_tp_processor(monkeypatch, joint, unbind, fused):
    monkeypatch.setenv("DIFFLET_BACKEND", "tpu")
    tp_sharding = importlib.import_module("difflet.models.flux.tp_sharding")
    sp = importlib.import_module("difflet.backends.tpu.flux.sp")
    monkeypatch.setattr(tp_sharding, "difflet_attention", _sdpa)

    torch.manual_seed(0)
    dim = HEADS * HEAD_DIM
    attn = FluxAttention(dim, heads=HEADS, dim_head=HEAD_DIM, bias=True,
                         added_kv_proj_dim=dim if joint else None, pre_only=not joint,
                         eps=1e-6).double().eval()
    for name, module in attn.named_modules():  # RMSNorm weights away from 1
        if name.startswith("norm") and getattr(module, "weight", None) is not None:
            module.weight.data.uniform_(0.5, 1.5)
    attn.processor = tp_sharding.FluxTPAttnProcessor(rope_unbind_dim=unbind)
    attn._difflet_fused_rope = fused

    img = torch.randn(2, IMAGE, dim, dtype=torch.float64)
    qkv = (attn.to_q(img), attn.to_k(img), attn.to_v(img))
    added = None
    if joint:
        txt = torch.randn(2, TEXT, dim, dtype=torch.float64)
        added = (attn.add_q_proj(txt), attn.add_k_proj(txt), attn.add_v_proj(txt))
    rope = _rope(IMAGE + (TEXT if joint else 0))

    want = attn.processor(sp._Precomputed(attn, qkv, added), hidden_states=qkv[0],
                          encoder_hidden_states=added[0] if joint else None,
                          image_rotary_emb=rope)
    got = sp._attention_heads_first(attn, qkv, added, rope, attention=_sdpa)
    atol = 1e-12
    if joint:  # the processor returns (image, text); ours is [text | image]
        img_want, txt_want = want
        torch.testing.assert_close(got[:, :TEXT], txt_want, rtol=0, atol=atol)
        torch.testing.assert_close(attn.to_out[1](got[:, TEXT:]), img_want, rtol=0, atol=atol)
    else:
        torch.testing.assert_close(got, want, rtol=0, atol=atol)
