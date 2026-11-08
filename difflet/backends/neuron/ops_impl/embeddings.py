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

``apply_rotary_emb_slices`` is the CPU op itself, for Wan's interleaved self-attention, whose q/k
come out of a bf16 matmul (a quantized q/k rotates in the half-split layout,
``difflet/models/wan/modeling_wan.py``): with ``x @ P`` two bf16 Wan 2.1 runs at tp4 480x832x9
gave two different clips (433 ms per step), with the stride-2 law the two are bit-identical and
the clip from before ``x @ P`` (438 ms). HunyuanVideo keeps ``apply_rotary_emb``: its bf16 runs
are reproducible with it, and its measured numbers are this op's.
"""

from __future__ import annotations

import torch

from difflet.backends.cpu.ops_impl.embeddings import apply_rotary_emb as apply_rotary_emb_slices


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


def apply_rotary_emb_fp32(hidden_states, freqs_cos, freqs_sin):
    """diffusers' fp32 interleaved law (the reference,
    ``difflet/backends/_torch_megatron/embeddings.py``: FLUX's call sites) with its
    ``reshape(..., 2).unbind`` / ``stack`` / ``flatten`` replaced by one matmul,
    ``x @ R`` with ``R[2i+1, 2i] = -1`` and ``R[2i, 2i+1] = 1``. That is diffusers' rotated
    operand (``-x[2i+1]``, ``x[2i]``), so the fp32 products and sum that follow are the
    reference's own, for any table dtype and any table values.

    ``x @ R`` has one +-1 term per output column, so it is exact for a bf16 ``x``, which FLUX
    feeds: the tensor engine multiplies bf16 by +-1, accumulates in fp32 and rounds a value that
    is already bf16. An fp32 ``x`` (a debug run) is exact only if neuronx-cc does not auto-cast
    the matmul to bf16; the neuron backend passes no ``--auto-cast`` flag, and the compiler's
    default casts fp32 matmuls, so that case is not claimed exact on the device (on the CPU it
    is)."""
    x = hidden_states
    d = x.shape[-1]
    idx = torch.arange(d, device=x.device)
    sign = torch.where(idx % 2 == 0, -1.0, 1.0)
    rot = ((idx[:, None] == (idx ^ 1)[None, :]).to(torch.float32) * sign[None, :]).to(x.dtype)
    return (x.float() * freqs_cos + (x @ rot).float() * freqs_sin).to(x.dtype)


__all__ = ["apply_rotary_emb", "apply_rotary_emb_fp32", "apply_rotary_emb_slices"]
