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
from difflet.backends.neuron.core.config_checks import (
    check_tp_matches,
    require_bf16_on_device,
    resolve_cp_degree,
    truncated_block_prefixes,
)
from difflet.backends.neuron.ops_impl.fp8_linear import _FP8LinearBase
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
    A ``num_layers`` below the checkpoint's (``NeuronWanConfig.from_pretrained(...,
    num_layers=N)``) loads the first N blocks and reports the others under
    ``load_report["skipped"]``.

    ``quant`` (a ``QuantSpec`` or its dict) is the base's MXFP8 wiring: the spec's target
    linears (``difflet.quant.targets``' ``"wan"`` table: the attention q/k/v and the FFN;
    every ``to_out.0``, the modulation tables, the embedders and ``proj_out`` stay bf16)
    become their FP8 twins on the meta module before the load, so ``model_path`` must then be
    the quantized checkpoint directory (``difflet/quant/checkpoint.py``) whose tensors those
    layers hold. The table is written in checkpoint-key spelling (``ffn.net.0.proj`` /
    ``ffn.net.2``) and the base matches it through :func:`checkpoint_key`, the same mapping
    the loader applies, so the FFN's ``net_in`` / ``net_out`` are quantized here exactly when
    the directory holds them quantized; the names in ``quantized_linears`` are the module's.
    No q/k/v fusion: ``WanAttention`` calls ``to_q``/``to_k``/``to_v`` directly. ``ffn.net.2``
    is row-parallel, quantized per rank K range for the launch's tp, so a spec for another tp
    is refused at construction. One spec serves both experts of Wan 2.2 (one block graph).
    A quantized self-attention q or k puts the module in the half-split rotary layout
    (``_after_load``). Trainium3 and compile mode only (decision D43,
    ``require_fp8_supported``).
    """

    block_attrs = ("blocks",)
    quant_model = "wan"
    quant_label = "Wan"

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
        check_tp_matches("NeuronWanConfig", config, self.parallel)
        # As for Qwen-Image: the config sizes the module and the parallel config the mesh, and
        # WanTransformer3DModel calls init_parallel_mesh(config) again under CP/CFG.
        config = resolve_cp_degree("NeuronWanConfig", config, self.parallel)
        config.validate()
        require_bf16_on_device("Wan", self.device.type, self.dtype)
        self.config = config
        # Blocks past num_layers are in the checkpoint on purpose when num_layers was cut.
        skipped = truncated_block_prefixes(config, "blocks.")
        if skipped:
            self.skipped_checkpoint_prefixes = skipped

    def build_module(self) -> torch.nn.Module:
        from difflet.models.wan.modeling_wan import WanTransformer3DModel

        return WanTransformer3DModel(self.config, dtype=self.dtype)

    def _after_load(self, module: torch.nn.Module) -> None:
        """When any self-attention's q or k is an MXFP8 twin, switch the whole module to the
        half-split rotary layout (``WanTransformer3DModel.use_half_split_rope_``) before the
        base prepares the twins, since the kernel path packs each weight in the order it then
        has. The interleaved rotary's pair swap on q/k from the MXFP8 kernel moves 2-byte
        elements (the 5x block of ``difflet.quant.targets``' Wan measurement); the half split
        moves whole rows. bf16 q/k keep the interleaved layout, faster for them. The rule is
        what is quantized, not the compute path, so ``dequant`` runs the layout the device
        does; one quantized q or k switches every self-attention, since the rotary tables are
        the module's. Each expert of Wan 2.2 is an application of its own, so each decides
        from its own module."""
        if any(isinstance(linear, _FP8LinearBase)
               for block in module.blocks for linear in (block.attn1.to_q, block.attn1.to_k)):
            reordered = module.use_half_split_rope_()
            if reordered and self.rank == 0:
                print(f"[wan] self-attention q/k reordered for the half-split rotary "
                      f"({reordered} attentions): their q/k come from the MXFP8 kernel",
                      flush=True)
        super()._after_load(module)

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
        """``(latents, timestep, text embeddings)`` at the static shape, in ``dtype`` except the
        timestep, which the DiT takes in fp32 as diffusers passes it.

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
            torch.zeros(batch, dtype=torch.float32),
            torch.zeros(batch, int(cfg.text_seq_len), int(cfg.text_dim), dtype=self.dtype),
        )


__all__ = ["NeuronWanTransformerApplication", "checkpoint_key"]
