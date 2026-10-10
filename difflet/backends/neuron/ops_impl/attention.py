"""Attention on the neuron device: the NKI flash kernel, with SDPA as the fallback.

Unmasked inference attention calls torch-neuronx's NKI flash-attention kernel
directly. Through ``F.scaled_dot_product_attention`` the kernel is reached only
when both sequence lengths are multiples of 512; the kernel itself takes other
lengths (Wan: 4,680 tokens), so the direct call avoids the decomposed path.
The kernel computes in bf16 internally, so only bf16/fp16 inputs take it; fp32 keeps
fp32 precision through SDPA. Masked or autograd calls, and shapes outside the
kernel's limits, also use SDPA.

The entry points keep the attention_cte calling contract that model code is
written against:

* layout flags: ``tp_q``/``tp_k`` True means q/k are ``[..., S, d]``; False means
  ``[..., d, S]``. ``v`` is always ``[..., S_k, d]``. ``tp_out`` True returns
  ``[..., d, S_q]`` instead of ``[..., S_q, d]``.
* ``scale=None`` means 1.0 (not SDPA's 1/sqrt(d)).
* ``bound_min``/``bound_max``: per query row, keys in [bound_min, bound_max) are
  valid; mutually exclusive with ``attention_mask``.
* boolean masks are True where attention is allowed; float masks are additive.

Context-parallel entry points are not implemented in phase 1 of this backend.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Kernel limits stated by torch-neuronx's SDPA gate, other than its 512-multiple lengths.
_KERNEL_DTYPES = (torch.bfloat16, torch.float16)
_KERNEL_MAX_HEAD_DIM = 128
_KERNEL_MAX_BATCH_HEADS = 512


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
    if kwargs:
        raise NotImplementedError(
            "neuron attention does not support attention_cte-only kwargs: "
            + ", ".join(sorted(kwargs))
        )
    q = q if tp_q else q.transpose(-1, -2)
    k = k if tp_k else k.transpose(-1, -2)

    if bound_min is not None or bound_max is not None:
        if bound_min is None or bound_max is None:
            raise ValueError("bound_min and bound_max must both be provided")
        if attention_mask is not None:
            raise ValueError("attention_mask and bounds are mutually exclusive")
        key_idx = torch.arange(k.shape[-2], device=k.device).view(1, 1, -1)
        attention_mask = (key_idx >= bound_min.to(key_idx.device)) & (
            key_idx < bound_max.to(key_idx.device)
        )

    if attention_mask is not None and causal:
        attention_mask = _merge_causal_mask(q, k, attention_mask)
        causal = False

    scale = 1.0 if scale is None else float(scale)
    if _can_use_flash_kernel(q, k, v, attention_mask):
        out = _flash_attention(q, k, v, scale=scale, causal=causal)
    else:
        # Pass SDPA's default scale when it matches, keeping the call on its plain path.
        sdpa_scale = None if math.isclose(scale, 1.0 / math.sqrt(q.shape[-1])) else scale
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=causal, scale=sdpa_scale
        )
    return out.transpose(-1, -2) if tp_out else out


def _can_use_flash_kernel(q, k, v, attention_mask) -> bool:
    if attention_mask is not None or not _on_neuron(q) or q.dtype not in _KERNEL_DTYPES:
        return False
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return False  # inference-only kernel call; autograd goes through SDPA
    if q.dim() not in (3, 4) or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        return False
    batch_heads = math.prod(q.shape[:-2])
    if q.shape[-1] > _KERNEL_MAX_HEAD_DIM or batch_heads > _KERNEL_MAX_BATCH_HEADS:
        return False
    return _flash_kernel() is not None


def _on_neuron(t) -> bool:
    return t.device.type == "neuron"


def _flash_attention(q, k, v, *, scale: float, causal: bool):
    """Run the kernel on [B, H, S, d] (3D [B*H, S, d] gets a unit batch axis)."""
    squeeze = q.dim() == 3
    if squeeze:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    out = _flash_kernel()(q, k, v, is_causal=causal, scale=scale, training=False)
    return out.squeeze(0) if squeeze else out


def _load_flash_kernel():
    try:
        from torch_neuronx.python_ops.nki_kernels.scaled_dot_product_attention import (
            scaled_dot_product_attention_kernel,
        )
    except ImportError:
        return None
    return scaled_dot_product_attention_kernel


# Resolved once at import: no import or cache lookup inside traced (compiled) code.
_FLASH_KERNEL = _load_flash_kernel()


def _flash_kernel():
    return _FLASH_KERNEL


def cross_attention(q, k, v, *, scale: float | None = None, attention_mask=None, **kwargs):
    return attention(
        q,
        k,
        v,
        scale=scale,
        causal=False,
        attention_mask=attention_mask,
        tp_q=False,
        tp_k=False,
        tp_out=False,
        **kwargs,
    )


def _merge_causal_mask(q, k, attention_mask):
    q_len, k_len = q.shape[-2], k.shape[-2]
    causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
    if attention_mask.dtype == torch.bool:
        return attention_mask & causal_mask
    causal_bias = torch.zeros((q_len, k_len), dtype=q.dtype, device=q.device)
    causal_bias = causal_bias.masked_fill(~causal_mask, torch.finfo(q.dtype).min)
    return attention_mask.to(device=q.device, dtype=q.dtype) + causal_bias


def _context_parallel_unsupported(name: str):
    def unsupported(*args, **kwargs):
        raise NotImplementedError(
            f"difflet.ops.{name} needs context parallelism, which the neuron backend "
            "does not implement yet; run with cp_degree=1"
        )

    unsupported.__name__ = name
    return unsupported


ring_attention = _context_parallel_unsupported("ring_attention")
joint_ring_attention = _context_parallel_unsupported("joint_ring_attention")
ulysses_attention = _context_parallel_unsupported("ulysses_attention")
joint_ulysses_attention = _context_parallel_unsupported("joint_ulysses_attention")

__all__ = [
    "attention",
    "cross_attention",
    "ring_attention",
    "joint_ring_attention",
    "ulysses_attention",
    "joint_ulysses_attention",
]
