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

import torch
import torch.nn.functional as F

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
        # Extension point for a fused kernel. torch_xla ships a TPU Pallas
        # flash attention (torch_xla.experimental.custom_kernel.flash_attention)
        # which would remove the score matrix entirely rather than merely
        # bounding it; it is unusable on this toolchain only because it pins
        # jax==0.7.1, which needs Python >= 3.11. Dropping it in here later is
        # a one-branch change, and chunking composes with it either way:
        # each block below is a normal SDPA call, so whatever lowering XLA
        # applies to SDPA still applies per block.
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
