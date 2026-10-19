"""Attention for the CUDA backend: ``torch.nn.functional.scaled_dot_product_attention``.

Contract (shared with the CPU oracle and the TPU backend, ``difflet/ops/attention.py``):

* ``scale=None`` means **1.0**, not SDPA's default ``1/sqrt(head_dim)`` — passing
  None through would silently rescale every score (plan Review Focus 2).
* ``attention_mask`` is bool (True = attend) or additive; ``bound_min``/``bound_max``
  (both or neither, exclusive with a mask) select a per-query key window, the
  ``attention_cte`` contract from Trainium.
* ``causal`` with an explicit mask folds the triangle into the mask, because SDPA
  rejects ``is_causal`` together with ``attn_mask``; additive masks use
  ``finfo.min`` rather than ``-inf`` so a fully masked row stays finite.
* ``tp_q``/``tp_k``/``tp_out`` are Trainium kernel hints: accepted and ignored.
* Inputs may be ``(B, H, S, D)`` or ``(B*H, S, D)``; SDPA accepts either batch shape.

Kernel selection (flash / cuDNN / efficient / math) is torch's; ``platform.
sdpa_kernel_report`` says which one a shape gets, recorded in Phase 0. No
``flash-attn`` dependency. The context-parallel modes raise (not supported yet).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_CP_DEFERRED = (
    "context-parallel attention is not supported on the CUDA backend yet; run with cp_degree=1"
)


def _bounds_to_mask(k, bound_min, bound_max, attention_mask):
    """attention_cte contract: per query row, keys in [bound_min, bound_max)."""
    if bound_min is None and bound_max is None:
        return attention_mask
    if bound_min is None or bound_max is None:
        raise ValueError("bound_min and bound_max must both be provided")
    if attention_mask is not None:
        raise ValueError("attention_mask and bounds are mutually exclusive")
    key_idx = torch.arange(k.shape[-2], device=k.device).view(1, 1, -1)
    return (key_idx >= bound_min.to(key_idx.device)) & (key_idx < bound_max.to(key_idx.device))


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
    attention_mask = _bounds_to_mask(k, bound_min, bound_max, attention_mask)
    scale = 1.0 if scale is None else scale

    # SDPA's fused kernels (flash / cuDNN / efficient) require 4-D (B, H, S, D); a 3-D
    # (B*H, S, D) call silently takes the math path, which materialises the S×S score
    # matrix per head (measured in Phase 0 on the Pro 6000: "flash" vs "math" for the
    # same tokens). Model code folds heads into the batch axis, so lift it here.
    folded = q.dim() == 3
    if folded:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    out = _sdpa(q, k, v, scale=scale, causal=causal, attention_mask=attention_mask)
    return out.squeeze(0) if folded else out


def _sdpa(q, k, v, *, scale: float, causal: bool, attention_mask):
    if attention_mask is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    q_len, k_len = q.shape[-2], k.shape[-2]
    if attention_mask.dtype == torch.bool:
        mask = attention_mask.to(q.device)
        if causal:
            tri = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
            mask = mask & tri
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

    additive = attention_mask.to(device=q.device, dtype=q.dtype)
    if causal:
        tri = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
        additive = additive.masked_fill(~tri, torch.finfo(q.dtype).min)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=additive, scale=scale)


def cross_attention(q, k, v, *, scale: float | None = None, attention_mask=None, **kwargs):
    kwargs.pop("tp_q", None)
    kwargs.pop("tp_k", None)
    kwargs.pop("tp_out", None)
    return attention(q, k, v, scale=scale, causal=False, attention_mask=attention_mask, **kwargs)


def ring_attention(*args, **kwargs):
    raise NotImplementedError(f"ring_attention: {_CP_DEFERRED}")


def joint_ring_attention(*args, **kwargs):
    raise NotImplementedError(f"joint_ring_attention: {_CP_DEFERRED}")


def ulysses_attention(*args, **kwargs):
    raise NotImplementedError(f"ulysses_attention: {_CP_DEFERRED}")


def joint_ulysses_attention(*args, **kwargs):
    raise NotImplementedError(f"joint_ulysses_attention: {_CP_DEFERRED}")


__all__ = [
    "attention",
    "cross_attention",
    "joint_ring_attention",
    "joint_ulysses_attention",
    "ring_attention",
    "ulysses_attention",
]
