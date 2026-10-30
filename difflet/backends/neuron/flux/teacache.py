"""TeaCache probe for the Flux DiT on the neuron backend.

``NeuronTeaCacheProbe`` over the DiT application's ``teacache_mod_input``, which reproduces the
Trainium fused probe's signal (block 0's modulated image input:
``transformer_blocks[0].norm1(x_embedder(x), time_text_embed(1000 t, 1000 g, pooled))[0]``), so
the same calibrations apply. Inputs arrive on the host from ``NeuronFluxPipeline``.
"""

from __future__ import annotations

from difflet.backends.neuron.core.teacache import NeuronTeaCacheProbe


class NeuronFluxTeaCacheProbe(NeuronTeaCacheProbe):
    """``teacache_delta(latents, timestep01, pooled, guidance)`` on the DiT application's model."""

    def __init__(self, dit) -> None:
        self.dit = dit

        def mod_input(hidden_states, timestep, pooled_projections, guidance):
            dev, dt = dit.device, dit.dtype
            g = guidance.to(dev, dt) if guidance is not None and guidance.numel() else None
            return dit.teacache_mod_input(hidden_states.to(dev, dt), timestep.to(dev, dt),
                                          pooled_projections.to(dev, dt), g)

        super().__init__(mod_input)


__all__ = ["NeuronFluxTeaCacheProbe"]
