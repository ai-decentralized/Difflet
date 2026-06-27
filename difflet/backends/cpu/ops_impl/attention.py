"""Torch-native attention for CPU numerical checks."""

from __future__ import annotations

import torch


def attention(
    q,
    k,
    v,
    *,
    scale: float | None = None,
    causal: bool = False,
    attention_mask=None,
    tp_q: bool = False,
    tp_k: bool = False,
    tp_out: bool = False,
    **kwargs,
):
    del tp_q, tp_k, tp_out, kwargs
    scale = 1.0 if scale is None else scale
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        q_len, k_len = scores.shape[-2:]
        mask = torch.ones((q_len, k_len), dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def cross_attention(q, k, v, *, scale: float | None = None, attention_mask=None, **kwargs):
    return attention(q, k, v, scale=scale, causal=False, attention_mask=attention_mask, **kwargs)


def ring_attention(q, k, v, *, scale: float, causal: bool = False):
    # Single-process CPU reference: cp_degree == 1, so the "ring" is just plain
    # non-causal attention over the local (== full) sequence.
    b, h, s_q, d = q.shape
    s_k = k.shape[2]
    out = attention(
        q.reshape(b * h, s_q, d),
        k.reshape(b * h, s_k, d),
        v.reshape(b * h, s_k, d),
        scale=scale,
        causal=causal,
        tp_q=True,
        tp_k=True,
        tp_out=False,
    )
    return out.reshape(b, h, s_q, d)


def joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale: float, causal: bool = False):
    # cp_degree == 1 reference: the ring degenerates to plain full joint attention
    # over the concatenated [image, text] keys. This is the merge-math oracle.
    full_k = torch.cat([image_k, text_k], dim=2)
    full_v = torch.cat([image_v, text_v], dim=2)
    b, h, s_q, d = q.shape
    s_k = full_k.shape[2]
    out = attention(
        q.reshape(b * h, s_q, d),
        full_k.reshape(b * h, s_k, d),
        full_v.reshape(b * h, s_k, d),
        scale=scale, causal=causal, tp_q=True, tp_k=True, tp_out=False,
    )
    return out.reshape(b, h, s_q, d)


__all__ = ["attention", "cross_attention", "ring_attention", "joint_ring_attention"]
