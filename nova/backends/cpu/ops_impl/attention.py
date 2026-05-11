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


__all__ = ["attention", "cross_attention"]
