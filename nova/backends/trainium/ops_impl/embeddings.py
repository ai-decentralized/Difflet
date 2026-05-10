"""Trainium positional embedding helpers."""

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
