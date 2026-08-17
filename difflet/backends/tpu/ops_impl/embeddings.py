"""Positional embedding helpers for TPU.

Identical math to the Trainium and CPU implementations — this is pure torch
with no backend-specific kernel, so it is reused verbatim rather than
rewritten. Keeping it byte-identical is deliberate: rotary embeddings are an
easy place for a silent interleaving bug, and divergence here would be
invisible until numerical validation.

Phase 2b of docs/plans/2026-08-16-tpu-backend-support.md.
"""

import torch


def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    original_dtype = hidden_states.dtype
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.to(original_dtype)


__all__ = ["apply_rotary_emb"]
