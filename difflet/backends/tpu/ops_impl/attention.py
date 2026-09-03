"""Attention for TPU.

Under tensor parallelism the heads are already split across ranks by the
surrounding column/row-parallel projections, so attention itself is purely
local — no collective belongs in here. That is why ``tp_q``/``tp_k``/
``tp_out`` are accepted and ignored, exactly as the CPU backend does; they are
kernel hints for Trainium's ``attention_cte``, not semantics.

Context-parallel modes (ring / ulysses / gather_kv) are deliberately NOT
implemented — they are Phase 5 work, and a wrong-but-silent CP implementation
is far worse than an explicit failure. They raise instead.

Phase 2c of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#: Query/key lengths must be padded to a multiple of this before the Pallas
#: flash kernel will accept them. 512 is the smallest value that worked at
#: Wan's 4680: 4736 (a multiple of 128, but not of 512) raises
#: ``broadcast_in_dim operand dimension sizes must either be 1, or ...``.
FLASH_BLOCK = 512

#: Below this many score elements, SDPA is *faster* than the fused kernel and
#: the kernel's fixed overhead dominates. Measured at q=4680, head_dim=128,
#: 10 heads, sweeping the key length (SDPA / flash, per call):
#:
#:      24 M  0.51 ms / 0.89 ms   0.58x   <- SDPA wins
#:      48 M  1.90 ms / 0.89 ms   2.14x
#:      96 M  3.49 ms / 0.91 ms   3.82x
#:     144 M  5.12 ms / 0.94 ms   5.42x
#:     219 M  7.99 ms / 1.19 ms   6.70x
#:
#: SDPA grows linearly with the score matrix because it materializes it; the
#: fused kernel is near-flat because it does not. The crossover sits between
#: 24 M and 48 M, so the threshold is set in the middle.
FLASH_MIN_SCORE_ELEMENTS = 32 * 1024 * 1024

#: Resolved lazily and cached: the ``custom_kernel`` module, or False.
_FLASH_KERNEL: object | None = None

#: Above this many score elements per attention call, compute the attention in
#: query blocks instead of one shot.
#:
#: SDPA on XLA materializes the full score matrix. At Qwen-Image 1024x1024 that
#: is 6 heads x 5120 x 5120 x 4 bytes = 600 MiB *per layer* in fp32 — measured,
#: as the exact size of the allocation that failed with "Attempting to allocate
#: 600.00M. There are 201.40M free." after the weight shard left ~6 GiB.
#:
#: The threshold is deliberately loose. The 600 MiB matrix above only failed
#: to fit because a missing torch.no_grad() was keeping every block's
#: intermediates alive; with that fixed, one-shot attention measures
#: 0.254 s/step against 0.313 s/step chunked at 1024 — chunking costs 24% and
#: is not needed at this size. It stays as the guard for genuinely larger
#: shapes, not as the normal path.
#:
#: The TPU Pallas flash-attention kernel would be better than either, removing
#: the matrix rather than bounding it, but torch_xla 2.9 pins it to jax==0.7.1,
#: which requires Python >= 3.11; this toolchain is on 3.10.
DEFAULT_MAX_SCORE_ELEMENTS = 512 * 1024 * 1024

#: Query rows per block once chunking kicks in. Peak score memory becomes
#: heads x QUERY_CHUNK x keys instead of heads x queries x keys.
QUERY_CHUNK = 1024

_CP_DEFERRED = (
    "context-parallel attention is not implemented on the TPU backend yet "
    "(Phase 5 of docs/plans/2026-08-16-tpu-backend-support.md); run with "
    "cp_degree=1"
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
    return (key_idx >= bound_min.to(key_idx.device)) & (
        key_idx < bound_max.to(key_idx.device)
    )


def _leading_size(tensor) -> int:
    """Product of everything before (queries, dim).

    Callers are not consistent about rank: difflet's model code folds heads
    into the batch axis and passes (B*heads, S, D), while a plain SDPA caller
    passes (B, heads, S, D). Assuming 4-D silently under-counted the score
    matrix by the head factor — 26 M elements instead of 157 M — so the
    threshold never fired on exactly the shape it was written for.
    """
    total = 1
    for dim in tensor.shape[:-2]:
        total *= int(dim)
    return total


def _should_chunk(q, k) -> bool:
    score_elements = _leading_size(q) * int(q.shape[-2]) * int(k.shape[-2])
    return score_elements > DEFAULT_MAX_SCORE_ELEMENTS


def _chunked_attention(q, k, v, *, scale: float, causal: bool):
    """Attention in query blocks, so the score matrix is never materialized whole.

    Each block is a complete, independent softmax over all keys, so this is
    exactly equal to the one-shot result — no online-softmax rescaling and no
    numerical difference, just a smaller peak allocation.

    It does not reduce compute. The same FLOPs run as more, smaller matmuls
    plus one concatenation, so expect it to be no faster and possibly slightly
    slower than a single call that fits. The point is that it fits.

    Causal masking still works because block ``i`` covers absolute query rows
    ``[start, start + rows)``, and the mask is built against those absolute
    positions rather than block-local ones.
    """
    queries = int(q.shape[-2])
    keys = int(k.shape[-2])
    outputs = []
    for start in range(0, queries, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, queries)
        block = q[..., start:stop, :]
        if causal:
            rows = torch.arange(start, stop, device=q.device).unsqueeze(-1)
            cols = torch.arange(keys, device=q.device).unsqueeze(0)
            outputs.append(
                F.scaled_dot_product_attention(
                    block, k, v, attn_mask=(cols <= rows), scale=scale
                )
            )
        else:
            outputs.append(
                F.scaled_dot_product_attention(block, k, v, scale=scale)
            )
    return torch.cat(outputs, dim=-2)


def _flash_kernel():
    """``torch_xla.experimental.custom_kernel``, or None when unusable.

    The kernel lowers through Pallas and therefore needs ``jax``. On Python
    3.10 there is no usable jax: ``pip install jax[tpu]`` resolves to 0.6.2 and
    downgrades libtpu out from under torch_xla. So this returns None on that
    toolchain and every caller falls back to SDPA — the backend stays correct,
    just slower. ``DIFFLET_TPU_FLASH=0`` forces the fallback for A/B testing.
    """
    global _FLASH_KERNEL
    if _FLASH_KERNEL is None:
        if os.environ.get("DIFFLET_TPU_FLASH", "1").lower() in ("0", "false", "no"):
            _FLASH_KERNEL = False
        else:
            try:
                import jax  # noqa: F401
                from torch_xla.experimental import custom_kernel

                _FLASH_KERNEL = custom_kernel
            except Exception as exc:  # noqa: BLE001
                logger.info("TPU flash attention unavailable (%s); using SDPA", exc)
                _FLASH_KERNEL = False
    return _FLASH_KERNEL or None


def _align(length: int) -> int:
    return ((int(length) + FLASH_BLOCK - 1) // FLASH_BLOCK) * FLASH_BLOCK


def _should_flash(q, k, causal: bool) -> bool:
    if causal:
        # Not a limitation of the kernel, of this wrapper: padding puts the
        # pad rows *after* the real ones, and reasoning about a causal
        # triangle over padded absolute positions is a correctness risk for
        # no gain — no diffusion DiT in this repo uses causal attention.
        return False
    head_dim = int(q.shape[-1])
    if head_dim != int(k.shape[-1]) or head_dim % 128:
        return False
    score_elements = _leading_size(q) * int(q.shape[-2]) * int(k.shape[-2])
    if score_elements < FLASH_MIN_SCORE_ELEMENTS:
        return False
    return _flash_kernel() is not None


def _flash_attention(q, k, v, *, scale: float):
    """Fused attention, padding the sequence to the kernel's block size.

    **The padding must be masked.** The kernel accepts an unaligned length
    without complaint and pads internally with zeros, which then take part in
    the softmax: at Wan's 4680 that silently returns a result with 5.4e-2
    relative error against an fp32 reference instead of 2.8e-3. Marking the pad
    rows as a separate segment is what makes it correct — and, measured, it
    costs nothing (1.18 ms padded-and-masked against 1.18 ms unmasked-wrong).
    """
    kernel = _flash_kernel()
    leading = _leading_size(q)
    seq_q, seq_k = int(q.shape[-2]), int(k.shape[-2])
    pad_q, pad_k = _align(seq_q), _align(seq_k)

    shape = q.shape
    q4 = q.reshape(1, leading, seq_q, q.shape[-1])
    k4 = k.reshape(1, leading, seq_k, k.shape[-1])
    v4 = v.reshape(1, leading, seq_k, v.shape[-1])

    q_ids = kv_ids = None
    if pad_q != seq_q or pad_k != seq_k:
        q4 = F.pad(q4, (0, 0, 0, pad_q - seq_q))
        k4 = F.pad(k4, (0, 0, 0, pad_k - seq_k))
        v4 = F.pad(v4, (0, 0, 0, pad_k - seq_k))
        q_ids = torch.zeros(1, pad_q, dtype=torch.int32, device=q.device)
        q_ids[:, seq_q:] = 1
        kv_ids = torch.zeros(1, pad_k, dtype=torch.int32, device=q.device)
        kv_ids[:, seq_k:] = 1

    out = kernel.flash_attention(
        q4, k4, v4, causal=False, sm_scale=float(scale),
        q_segment_ids=q_ids, kv_segment_ids=kv_ids,
    )
    return out[:, :, :seq_q, :].reshape(shape[:-1] + (v.shape[-1],))


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
    # NOTE: the op contract treats scale=None as 1.0, NOT as SDPA's default
    # 1/sqrt(head_dim). Passing None through to SDPA would silently rescale
    # every score.
    scale = 1.0 if scale is None else scale

    if attention_mask is None:
        # The fused kernel removes the score matrix rather than merely
        # bounding it, so it comes first and makes chunking moot when it
        # applies. Measured on a v5e at Wan's self-attention shape (10 heads,
        # 4680): SDPA 7.99 ms at 7.1% MFU against 1.19 ms at 47.9%, and the
        # fused result is *closer* to an fp32 reference (2.8e-3 vs 3.3e-3)
        # because it keeps the softmax statistics in fp32 without ever
        # materializing the matrix. It declines small shapes and any causal or
        # masked call; see _should_flash.
        if _should_flash(q, k, causal):
            return _flash_attention(q, k, v, scale=scale)
        if _should_chunk(q, k):
            return _chunked_attention(q, k, v, scale=scale, causal=causal)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    # SDPA rejects is_causal together with an explicit mask, so fold the causal
    # triangle into the mask when both are present.
    if attention_mask.dtype == torch.bool:
        mask = attention_mask
        if causal:
            q_len, k_len = q.shape[-2], k.shape[-2]
            tri = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
            mask = mask & tri
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

    additive = attention_mask.to(device=q.device, dtype=q.dtype)
    if causal:
        q_len, k_len = q.shape[-2], k.shape[-2]
        tri = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
        additive = additive.masked_fill(~tri, torch.finfo(q.dtype).min)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=additive, scale=scale)


def cross_attention(q, k, v, *, scale: float | None = None, attention_mask=None, **kwargs):
    return attention(
        q, k, v, scale=scale, causal=False, attention_mask=attention_mask, **kwargs
    )


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
