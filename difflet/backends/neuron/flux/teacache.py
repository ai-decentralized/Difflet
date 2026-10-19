"""TeaCache probe for the Flux DiT on the neuron backend.

The controller (``difflet/pipeline/teacache.py``) and the denoising loop that consults it
(``NeuronFluxPipeline._call_with_teacache``) are backend-neutral; only the probe is not. It
reproduces the Trainium fused probe's signal exactly, so the same calibrations apply:

    m_t = block_0.norm1(x_embedder(x_t), time_text_embed(1000 t, 1000 g, pooled))[0]
    delta_t = mean|m_t - m_{t-1}| / (mean|m_{t-1}| + 1e-8)        (in float32)

with ``m_{t-1}`` kept on the device in the model's dtype and starting at zero. ``norm1``'s
modulation is a column-parallel linear with its output gathered, so ``m_t`` is whole on every
rank and every rank computes the same ``delta_t`` (no collective).
"""

from __future__ import annotations

import torch


class NeuronFluxTeaCacheProbe:
    """``teacache_delta(latents, timestep01, pooled, guidance)`` on the DiT application's model."""

    #: Read by NeuronFluxPipeline: the probe returns the signal itself (the fused interface).
    teacache_probe_fused = True

    def __init__(self, dit) -> None:
        self.dit = dit
        self._prev: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev = None

    @torch.no_grad()
    def teacache_delta(self, hidden_states, timestep, pooled_projections, guidance):
        dev, dt = self.dit.device, self.dit.dtype
        m = self.dit.teacache_mod_input(
            hidden_states.to(dev, dt), timestep.to(dev, dt), pooled_projections.to(dev, dt),
            guidance.to(dev, dt) if guidance is not None and guidance.numel() else None)
        prev = self._prev if self._prev is not None else torch.zeros_like(m)
        mf, pf = m.float(), prev.float()
        delta = (mf - pf).abs().mean() / (pf.abs().mean() + 1e-8)
        self._prev = m
        return delta


__all__ = ["NeuronFluxTeaCacheProbe"]
