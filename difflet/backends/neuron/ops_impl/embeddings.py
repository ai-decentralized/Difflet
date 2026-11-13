"""Rotary embedding for the neuron device: the CPU backend's interleaved-pair law
(``x[2i], x[2i+1]`` rotated by ``freqs[i]``), computed without strided slices of the
activation.

The CPU op (``difflet/backends/cpu/ops_impl/embeddings.py``) splits the pairs with
``unflatten(-1, (-1, 2)).unbind(-1)`` and writes the result back with ``out[..., 0::2]`` /
``out[..., 1::2]``. When ``x`` comes out of a bf16 matmul the compiler lays the matmul's
output out for that stride-2 access; when it comes out of an opaque custom op (the MXFP8
``matmul_mxfp8`` kernel) it cannot, and the deinterleave becomes element-wise: Wan 2.1's
self-attention with only ``to_q`` quantized ran a 2409 ms step against 438 ms in bf16
(2026-11-07, ``docs/verification/2026-11-07-mxfp8-video-models.md``), while cross-attention,
which has no RoPE, was unaffected. Here the pair swap is a matmul with a constant
permutation matrix (``x @ P``, ``P[2i+1, 2i] = P[2i, 2i+1] = 1``), which every layout feeds
the tensor engine without a gather, and the rotation is two element-wise products and one
add per element, so the result is bit-identical to the CPU op: ``+-x`` pass through the
matmul's fp32 accumulation exactly, ``a - b*s`` equals ``a + b*(-s)``, and the addition is
commutative. The constants are built from ``arange`` inside the op (no module-level cache:
Dynamo traces one graph per block class, and the compiler folds them).
"""

from __future__ import annotations

import torch


def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    original_dtype = hidden_states.dtype
    d = hidden_states.shape[-1]
    idx = torch.arange(d, device=hidden_states.device)
    # P[j, i] = 1 where j = i ^ 1: (x @ P)[..., i] = x[..., i ^ 1], the pair's other element.
    swap = (idx[:, None] == (idx ^ 1)[None, :]).to(hidden_states.dtype)
    # The swapped neighbour's sine enters with - on even positions and + on odd ones.
    sign = torch.where(idx % 2 == 0, -1.0, 1.0).to(freqs_sin.dtype)
    # cos / sin per element: each pair's value on both of its positions (the tables are
    # small and constant per step; these slices are not on the activation).
    cos = freqs_cos[..., 0::2].repeat_interleave(2, dim=-1)
    sin = freqs_sin[..., 1::2].repeat_interleave(2, dim=-1) * sign
    swapped = hidden_states @ swap
    out = hidden_states * cos + swapped * sin
    return out.to(original_dtype)


__all__ = ["apply_rotary_emb"]
