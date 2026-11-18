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


# diffusers' own lines, the ops FLUX's call sites traced before they were lifted into an op:
# the Neuron AoT graph is unchanged.
from difflet.backends._torch_megatron.embeddings import apply_rotary_emb_fp32  # noqa: E402,F401

__all__ = ["apply_rotary_emb", "apply_rotary_emb_fp32"]
