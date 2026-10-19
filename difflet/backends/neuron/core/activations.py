"""Activations the neuron backend evaluates differently from upstream.

``Fp32GELU`` is a feed-forward's input projection and tanh (or exact) GELU in diffusers'
``GELU`` layout, with the activation evaluated in fp32 and cast back to the projection's dtype.
Why: on trn2 the device's bf16 ``gelu(approximate="tanh")`` is less accurate than the CPU's
(1.0e-3 rel_l1 from fp32 against 6.9e-4, one NeuronCore, eager, 2310 x 4096 inputs of std 3),
which put LTX-2's 4-block preflight at 1.54x the diffusers bf16 control's error where the gate is
1.15x; in fp32 it is 1.05x, at the cost of one fp32 elementwise pass over the FFN's local
intermediate per block (``difflet/backends/neuron/ltx_2/transformer.py``, ``_FP32_GELU``).

It is a module of its own rather than diffusers' ``GELU`` with its class swapped at runtime: it
holds the same ``proj`` (so the state dict keys stay ``ff.net.0.proj.*``, and a TP-sharded
``proj`` stays sharded) and depends on nothing else of ``GELU``'s attribute layout. LTX-2 uses
it. Wan's FFN (``F.gelu`` in ``modeling_wan.py``) and Qwen-Image's ``img_mlp``/``txt_mlp``
(diffusers' ``GELU``) run the same device op and keep it in bf16 until the op probe has been
repeated on their own activations; ``use_fp32_gelu`` is how a model opts in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Fp32GELU(torch.nn.Module):
    """``proj`` then GELU in fp32, cast back to ``proj``'s output dtype; diffusers' ``GELU``
    computes the same thing in the input dtype (its ``forward`` is ``gelu(proj(x))``)."""

    def __init__(self, proj: torch.nn.Module, approximate: str = "none") -> None:
        super().__init__()
        self.proj = proj
        self.approximate = approximate

    @classmethod
    def from_gelu(cls, gelu: torch.nn.Module) -> Fp32GELU:
        """The same projection module (not a copy) and approximation as diffusers' ``gelu``."""
        return cls(gelu.proj, gelu.approximate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states.float(), approximate=self.approximate).to(hidden_states.dtype)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


def use_fp32_gelu(feed_forward: torch.nn.Module) -> bool:
    """Replace a diffusers ``FeedForward``'s ``GELU`` (``net[0]``) with ``Fp32GELU`` in place;
    False, and nothing changed, when ``net[0]`` is not exactly diffusers' ``GELU`` (another
    activation, or already replaced)."""
    from diffusers.models.activations import GELU

    activation = feed_forward.net[0]
    if type(activation) is not GELU:
        return False
    feed_forward.net[0] = Fp32GELU.from_gelu(activation)
    return True


__all__ = ["Fp32GELU", "use_fp32_gelu"]
