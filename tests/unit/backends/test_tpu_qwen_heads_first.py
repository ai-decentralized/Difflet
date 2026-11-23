"""Qwen-Image's heads-first split-halves attention (backends/tpu/qwen_image/sp.py)
equals the shared processor it replaces (models/qwen_image/modeling_qwen_image
._QwenImageTrainiumAttnProcessor), on the CPU.

A small Qwen-shaped attention (q/k/v and added text projections, per-head
RMSNorms, two output projections), float64 weights; the reference runs the
shared processor on the stock weights, the new path permutes q/k head channels
(``install_split_rope_``) and runs ``_attention_heads_first``. Both attend
through the same SDPA. The shared RoPE computes in float32
(``_apply_qwen_rope_real``), as ours does, so they agree to float32 rounding.
"""

from __future__ import annotations

import copy
import importlib
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("diffusers")
from diffusers.models.normalization import RMSNorm  # noqa: E402

HEADS, HEAD_DIM, TEXT, IMAGE = 3, 16, 5, 7


def _sdpa(q, k, v, *, scale, causal, **_):
    return F.scaled_dot_product_attention(q[None], k[None], v[None], scale=scale,
                                          is_causal=causal)[0]


def _attn():
    dim = HEADS * HEAD_DIM
    attn = nn.Module()
    attn.heads = HEADS
    for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        setattr(attn, name, nn.Linear(dim, dim))
    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        norm = RMSNorm(HEAD_DIM, eps=1e-6)
        norm.weight.data.uniform_(0.5, 1.5)
        setattr(attn, name, norm)
    attn.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
    attn.to_add_out = nn.Linear(dim, dim)
    return attn.double().eval()


def _rope(seq):  # the real (cos, sin) pair, each value repeated for its pair
    angles = torch.randn(seq, HEAD_DIM // 2, dtype=torch.float64)
    return angles.cos().repeat_interleave(2, -1), angles.sin().repeat_interleave(2, -1)


def test_heads_first_matches_shared_processor(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "tpu")
    modeling = importlib.import_module("difflet.models.qwen_image.modeling_qwen_image")
    sp = importlib.import_module("difflet.backends.tpu.qwen_image.sp")
    monkeypatch.setattr(modeling, "attention", _sdpa)

    torch.manual_seed(0)
    attn = _attn()
    img = torch.randn(2, IMAGE, HEADS * HEAD_DIM, dtype=torch.float64)
    txt = torch.randn(2, TEXT, HEADS * HEAD_DIM, dtype=torch.float64)
    rope = (_rope(IMAGE), _rope(TEXT))
    want_img, want_txt = modeling._QwenImageTrainiumAttnProcessor()(
        attn, img, encoder_hidden_states=txt, image_rotary_emb=rope)

    ours = copy.deepcopy(attn)
    sp.install_split_rope_([types.SimpleNamespace(attn=ours)])
    img_qkv = (ours.to_q(img), ours.to_k(img), ours.to_v(img))
    txt_qkv = (ours.add_q_proj(txt), ours.add_k_proj(txt), ours.add_v_proj(txt))
    img_ctx, txt_ctx = sp._attention_heads_first(ours, img_qkv, txt_qkv, rope, attention=_sdpa)
    torch.testing.assert_close(ours.to_out[0](img_ctx), want_img, rtol=0, atol=1e-5)
    torch.testing.assert_close(ours.to_add_out(txt_ctx), want_txt, rtol=0, atol=1e-5)


def test_permutation_moves_pairs_to_halves():
    sp = importlib.import_module("difflet.backends.tpu.qwen_image.sp")
    assert sp.pair_split_permutation(8).tolist() == [0, 2, 4, 6, 1, 3, 5, 7]
