"""Rotary embedding: the CPU backend's pure-torch implementation, verified on the neuron device.

``apply_rotary_emb`` (what ``difflet.ops.apply_rotary_emb`` resolves to on this backend) is the
CPU backend's function itself. ``apply_rotary_emb_stacked`` is the same math without its
strided writes, for a model that opts in on this backend (today Wan's DiT:
``difflet/backends/neuron/wan/compile_options.py``).
"""

import torch

from difflet.backends.cpu.ops_impl.embeddings import apply_rotary_emb


def apply_rotary_emb_stacked(hidden_states, freqs_cos, freqs_sin):
    """``apply_rotary_emb`` with the interleaved result built by ``stack`` + ``flatten``.

    The shared helper writes the even and odd lanes with ``out[..., 0::2] = ...`` and
    ``out[..., 1::2] = ...``; functionalized, those are two ``slice_scatter`` ops per call, which
    torch-neuronx lowers to ``stablehlo.scatter``. Stacking the two halves on a new last axis and
    flattening it interleaves them the same way. The products and sums are the shared helper's,
    in the promoted dtype, with one rounding to the input dtype, so the result is the same bits
    on CPU (tests/unit/backends/test_neuron_wan_step_levers.py).
    """
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.to(hidden_states.dtype)


__all__ = ["apply_rotary_emb", "apply_rotary_emb_stacked"]
