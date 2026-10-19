"""Neuron application for one Wan DiT expert.

The TPU counterpart is ``TpuWanTransformerApplication`` (``difflet/backends/tpu/wan/
transformer.py``); both wrap the same backend-neutral ``WanTransformer3DModel``
(``difflet/models/wan/modeling_wan.py``). Here the lifecycle is ``TorchNeuronApplicationBase``'s:
the module is built on ``meta`` after the TP mesh exists (so ``WanAttention`` shards the heads),
each rank loads its own slices of the diffusers checkpoint in ``dtype``, and in compile mode the
identical ``WanTransformerBlock``s in ``blocks`` are compiled in place into one shared graph while
the top level (Conv3d patch embedding, condition embedder, rotary tables, final norm and
projection) stays eager. The warm-up runs at the one static shape of ``NeuronWanConfig``.

"One expert" is the unit, as on TPU: Wan2.1 has one DiT; Wan2.2 A14B's second expert would be a
second instance of this class.
"""

from __future__ import annotations

import re

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.backends.neuron.wan.config import NeuronWanConfig

# The module -> checkpoint direction of the FFN renames (copied from
# difflet/backends/tpu/wan/transformer.py:45-48): difflet's WanFeedForward names the
# projections net_in/net_out where diffusers wraps them in a ModuleList.
_MODULE_TO_CHECKPOINT: list[tuple[str, str]] = [
    (r"\.ffn\.net_in\.", ".ffn.net.0.proj."),
    (r"\.ffn\.net_out\.", ".ffn.net.2."),
]


def checkpoint_key(name: str) -> str:
    """Diffusers checkpoint key holding the module parameter ``name``."""
    for pattern, replacement in _MODULE_TO_CHECKPOINT:
        name = re.sub(pattern, replacement, name)
    return name


class NeuronWanTransformerApplication(TorchNeuronApplicationBase):
    """One Wan DiT expert, sharded across the TP group, at the config's static shape.

    ``model_path`` is the expert's directory (``<model>/transformer``); ``config.tp_degree``
    must equal ``parallel.tp_degree`` (the module is sized by the config, the mesh by
    ``parallel``). On the neuron device the dtype must be ``torch.bfloat16`` (decision D42:
    Wan passes ``reduce_dtype=dtype`` to its TP linears and the neuron layers ignore it,
    all-reducing in the activation dtype; that matches NxD for the bf16 runs the ruling was
    made for, and this check stops any other dtype from reaching the device unexamined).
    """

    block_attrs = ("blocks",)

    def __init__(
        self,
        *,
        model_path,
        config: NeuronWanConfig,
        parallel=None,
        dtype=torch.bfloat16,
        exec_mode: str | None = None,
        device="neuron",
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            exec_mode=exec_mode,
            device=device,
            **kwargs,
        )
        config.validate()
        if int(config.tp_degree) != int(self.parallel.tp_degree):
            raise ValueError(
                f"NeuronWanConfig has tp_degree={config.tp_degree} but the parallel config has "
                f"tp_degree={self.parallel.tp_degree}; the module would be sharded for one and "
                "the mesh built for the other"
            )
        if self.device.type == "neuron" and self.dtype is not torch.bfloat16:
            raise ValueError(
                f"the Wan DiT runs in torch.bfloat16 on the neuron device, got {self.dtype} "
                "(decision D42: the TP layers all-reduce in the activation dtype, checked "
                "against NxD for bf16 only)"
            )
        self.config = config

    def build_module(self) -> torch.nn.Module:
        from difflet.models.wan.modeling_wan import WanTransformer3DModel

        return WanTransformer3DModel(self.config, dtype=self.dtype)

    def checkpoint_key(self, name: str) -> str:
        return checkpoint_key(name)

    @torch.no_grad()
    def teacache_mod_input(self, hidden_states: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """The TeaCache signal, as the model defines it (``WanTransformer3DModel.teacache_mod_input``,
        the method the Trainium CPU shadow runs): block 0's modulated self-attention input. Its
        modulation is timestep-only; the prompt embeddings are an input only because
        ``condition_embedder`` projects them (unused here), so a fixed zero tensor of the right
        shape stands in. Host inputs; moved to this rank's device in the model's dtype."""
        model = self.module
        if getattr(self, "_teacache_text", None) is None:
            text_dim = getattr(getattr(model, "config", None), "text_dim", None)
            if text_dim is None:
                text_dim = next(m for m in model.condition_embedder.modules()
                                if isinstance(m, torch.nn.Linear)).in_features
            self._teacache_text = torch.zeros(int(self.config.batch_size), int(self.config.text_seq_len),
                                              int(text_dim), dtype=self.dtype, device=self.device)
        return model.teacache_mod_input(hidden_states.to(self.device, self.dtype),
                                        timestep.to(self.device), self._teacache_text)

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """``(latents, timestep, text embeddings)`` at the static shape, in ``dtype``.

        These are exactly the per-step shapes the orchestrator sends (two batch-1 forwards per
        CFG step), so the warm-up compiles the only block graph the run needs. With CFG
        parallelism the orchestrator sends one batch of 2 ([uncond, cond]) and the model splits it
        over the CFG group, so each rank's blocks still see batch 1.
        """
        cfg = self.config
        batch = 2 if cfg.cfg_parallel_enabled else int(cfg.batch_size)
        return (
            torch.zeros(
                batch,
                int(cfg.in_channels),
                cfg.latent_frames,
                cfg.latent_height,
                cfg.latent_width,
                dtype=self.dtype,
            ),
            torch.zeros(batch, dtype=self.dtype),
            torch.zeros(batch, int(cfg.text_seq_len), int(cfg.text_dim), dtype=self.dtype),
        )


__all__ = ["NeuronWanTransformerApplication", "checkpoint_key"]
