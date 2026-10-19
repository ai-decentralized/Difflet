"""TeaCache probe on the neuron backend, shared by the models.

The TeaCache controller (``difflet/pipeline/teacache.py``) and each model's denoising loop are
backend-neutral; the probe that produces the per-step signal is not (the Trainium path compiles
a separate probe program). Here the probe runs on the loaded DiT's own modules: a model provides
``mod_input(*inputs)`` (its block-0 modulated input, as its Trainium probe defines it), and

    delta_t = mean|m_t - m_{t-1}| / (mean|m_{t-1}| + 1e-8)        (float32)

with ``m_{t-1}`` kept on the device in the model's dtype, starting at zero. Each model's
``mod_input`` is whole on every rank (gathered modulation, full-sequence prefix), so every rank
computes the same ``delta_t`` without a collective.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class NeuronTeaCacheProbe:
    """``teacache_delta(*inputs)`` -> 0-dim fp32 tensor on the device."""

    #: Read by NeuronFluxPipeline: the probe returns the signal itself (the fused interface).
    teacache_probe_fused = True

    def __init__(self, mod_input: Callable[..., torch.Tensor]) -> None:
        self.mod_input = mod_input
        self._prev: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev = None

    @torch.no_grad()
    def teacache_delta(self, *inputs):
        m = self.mod_input(*inputs)
        prev = self._prev if self._prev is not None else torch.zeros_like(m)
        mf, pf = m.float(), prev.float()
        delta = (mf - pf).abs().mean() / (pf.abs().mean() + 1e-8)
        self._prev = m
        return delta


__all__ = ["NeuronTeaCacheProbe"]
