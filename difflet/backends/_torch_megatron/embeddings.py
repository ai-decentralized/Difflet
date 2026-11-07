"""Positional embedding helpers for the torch backends (TPU, GPU).

Identical math to the Trainium and CPU implementations — this is pure torch
with no backend-specific kernel, so it is reused verbatim rather than
rewritten. Keeping it byte-identical is deliberate: rotary embeddings are an
easy place for a silent interleaving bug, and divergence here would be
invisible until numerical validation.
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


def apply_rotary_emb_fp32(hidden_states, freqs_cos, freqs_sin):
    """diffusers' ``apply_rotary_emb(use_real=True, use_real_unbind_dim=-1)`` on a 4-D
    ``hidden_states`` with tables already broadcast to it: the interleaved pairs rotated in fp32
    and cast back. This is the reference, diffusers' own lines in diffusers' order, so every
    backend that exports it traces the ops FLUX's call sites ran before the op existed (the
    Trainium AoT FLUX graph is unchanged); the neuron backend owns a matmul form of it."""
    x = hidden_states
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * freqs_cos + x_rotated.float() * freqs_sin).to(x.dtype)


__all__ = ["apply_rotary_emb", "apply_rotary_emb_fp32"]
