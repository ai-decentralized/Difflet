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
    bound_min=None,
    bound_max=None,
    tp_q: bool = False,
    tp_k: bool = False,
    tp_out: bool = False,
    **kwargs,
):
    del tp_q, tp_k, tp_out, kwargs
    if bound_min is not None or bound_max is not None:
        # attention_cte contiguous-bounds contract: per query row, keys in
        # [bound_min, bound_max) are valid. Silently ignoring these ran the
        # attention UNMASKED on CPU (garbage padding keys poisoned every query).
        if bound_min is None or bound_max is None:
            raise ValueError("bound_min and bound_max must both be provided")
        if attention_mask is not None:
            raise ValueError("attention_mask and bounds are mutually exclusive")
        key_idx = torch.arange(k.shape[-2], device=k.device).view(1, 1, -1)
        attention_mask = (key_idx >= bound_min.to(key_idx.device)) & (
            key_idx < bound_max.to(key_idx.device)
        )
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


def ulysses_attention(q, k, v, *, scale: float, causal: bool = False):
    # Single-process CPU reference: cp_degree == 1, so both all-to-alls are
    # identity and Ulysses is plain attention over the local (== full) sequence.
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


def joint_ulysses_attention(
    q_img, q_txt, image_k, image_v, text_k, text_v, *, scale: float, causal: bool = False,
    key_valid_len=None,
):
    # cp_degree == 1 reference: the all-to-alls degenerate to identity, so this is
    # plain full joint attention over the concatenated [image, text] keys. The two
    # streams are split back apart on the way out, matching the device contract.
    full_q = torch.cat([q_img, q_txt], dim=2)
    full_k = torch.cat([image_k, text_k], dim=2)
    full_v = torch.cat([image_v, text_v], dim=2)
    b, h, s_q, d = full_q.shape
    s_k = full_k.shape[2]
    bounds = {}
    if key_valid_len is not None:
        # Same contract as the device path: keys [0, count) valid for every query.
        count = key_valid_len.to(torch.int32).reshape(b, 1, 1, 1)
        bound_max = count.expand(b, h, s_q, 1).reshape(b * h, s_q, 1).contiguous()
        bounds = {"bound_min": torch.zeros_like(bound_max), "bound_max": bound_max}
    out = attention(
        full_q.reshape(b * h, s_q, d),
        full_k.reshape(b * h, s_k, d),
        full_v.reshape(b * h, s_k, d),
        scale=scale, causal=causal, tp_q=True, tp_k=True, tp_out=False, **bounds,
    )
    out = out.reshape(b, h, s_q, d)
    s_img = q_img.shape[2]
    return out[:, :, :s_img], out[:, :, s_img:]


__all__ = [
    "attention",
    "cross_attention",
    "ring_attention",
    "joint_ring_attention",
    "ulysses_attention",
    "joint_ulysses_attention",
]
