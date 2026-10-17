"""Neuron application for the Qwen-Image DiT.

The TPU counterpart is ``TpuQwenImageTransformerApplication`` (``difflet/backends/tpu/
qwen_image/transformer.py``) and the Trainium one shares this class's name in
``difflet/backends/trainium/qwen_image/transformer.py``; all three wrap the same backend-neutral
``_QwenImageTransformerTraceModule`` (``difflet/models/qwen_image/modeling_qwen_image.py``),
which nests diffusers' ``QwenImageTransformer2DModel`` as ``self.transformer``, swaps its large
linears for ``difflet.ops`` TP layers, replaces the complex RoPE with static real tables for the
one packed shape and installs the double-stream attention processor (gather_kv / ring / ulysses
under CP). Here the lifecycle is ``TorchNeuronApplicationBase``'s: the module is built on
``meta`` after the mesh exists (so the TP linears size their shards and the CP processor captures
its group), each rank loads its own slices of the diffusers checkpoint in ``dtype``, and in
compile mode the identical ``QwenImageTransformerBlock``s are compiled in place into one shared
graph while the top level (patch/text input projections, timestep embedding, final norm and
projection, and the CP scatter/gather) stays eager.

Sequence parallelism (``sp_enabled``, Megatron SP within the TP group) builds the trace module's
``QwenImageSPTransformer2DModel`` fork (``difflet/models/qwen_image/modeling_qwen.py``) instead
of diffusers' class: the same submodule layout and checkpoint keys, the four row-parallel
outputs built with ``reduce_output=False``, both residual streams sharded along their own
sequence axis between the blocks, and each TP region entered with an all-gather and left with a
reduce-scatter instead of the all-reduce. The neuron ops supply those collectives over the TP
group (``ops_impl/collectives.py``); the fork's ``_sp_unbias`` removes the ``tp - 1`` extra
biases the un-reduced partials carry (the neuron ``RowParallelLinear`` adds the full bias to
each partial, as NxD's does). The blocks' collectives are functional collectives, so the block
graph compiles as one graph per block class, as without SP.

The text mask is dropped, as on Trainium and TPU: prompts are zero-padded to ``text_seq_len``
and attended unmasked. The modeling's forward deletes ``encoder_hidden_states_mask`` (the
diffusers mask path builds the active length with ``arange``/``where``, which the fixed-shape
traces could not take); the padded rows are zeros before ``txt_norm``/``txt_in``, not absent, so
this matches diffusers only where the mask is all ones, and it is the semantics every Difflet
Qwen-Image backend shares. Masked attention is a follow-up, not a phase-1 feature.
"""

from __future__ import annotations

import dataclasses

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.backends.neuron.qwen_image.config import NeuronQwenImageConfig

#: The trace module nests the diffusers model under this attribute; checkpoint keys are the
#: diffusers model's own (``transformer_blocks.0.img_mod.1.weight``, ``norm_out.linear.bias``).
CHECKPOINT_PREFIX = "transformer."


def checkpoint_key(name: str) -> str:
    """Diffusers checkpoint key holding the module parameter ``name``.

    Only the wrapper prefix differs: the TP replacements keep diffusers' attribute paths
    (``img_mod[1]``, ``img_mlp.net[0].proj``, ``attn.to_out[0]``, ``norm_out.linear``,
    ``time_text_embed.timestep_embedder.linear_1``), and the wrapper's own ``global_rank``
    (``SPMDRank``) holds no state on this backend.
    """
    return name.removeprefix(CHECKPOINT_PREFIX)


def _trace_module_class():
    """``_QwenImageTransformerTraceModule`` with the mask and guidance optional.

    The warm-up broadcasts tensors only (``broadcast_tensor`` has no None), so it calls the
    module with the three tensors the DiT actually consumes; the orchestrator's five-argument
    call keeps working. Built lazily: importing the modeling binds ``difflet.ops`` to the
    selected backend, which must not happen when this module is merely imported.
    """
    from difflet.models.qwen_image.modeling_qwen_image import _QwenImageTransformerTraceModule

    class NeuronQwenImageTraceModule(_QwenImageTransformerTraceModule):
        def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_mask: torch.Tensor | None = None,
            guidance: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return super().forward(
                hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask,
                guidance,
            )

    return NeuronQwenImageTraceModule


class NeuronQwenImageTransformerApplication(TorchNeuronApplicationBase):
    """The Qwen-Image DiT, sharded across the TP group (and the image tokens across the CP
    group), at the config's static shape.

    ``model_path`` is the transformer directory (``<model>/transformer``). The config sizes the
    module and the parallel config the mesh, so they must agree: ``tp_degree`` equal,
    ``context_parallel_enabled`` iff ``parallel.cp_degree > 1`` with the same ``cp_mode``,
    ``sp_enabled`` equal, and no CFG parallelism (Qwen-Image is guidance-distilled). With CP on, the trace module calls
    ``init_parallel_mesh(config)`` again while it is built; the neuron mesh derives cp as
    ``world / (tp * cfg * dp)`` from the config, which these checks make equal to the spec the
    base initialised from ``parallel``, so that call is the idempotent no-op. ``config.cp_degree``
    left None is filled from ``parallel`` (on a copy) so ``validate()`` runs the cp checks.

    On the neuron device the dtype must be ``torch.bfloat16`` (decision D42, as for Wan: the
    neuron TP linears all-reduce in the activation dtype, checked against NxD for bf16 only).
    A ``num_layers`` below the checkpoint's (``NeuronQwenImageConfig.from_pretrained(...,
    num_layers=N)``) loads the first N blocks and reports the others under
    ``load_report["skipped"]``.
    """

    block_attrs = ("transformer.transformer_blocks",)

    def __init__(
        self,
        *,
        model_path,
        config: NeuronQwenImageConfig,
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
        parallel = self.parallel
        if int(config.tp_degree) != int(parallel.tp_degree):
            raise ValueError(
                f"NeuronQwenImageConfig has tp_degree={config.tp_degree} but the parallel config "
                f"has tp_degree={parallel.tp_degree}; the module would be sharded for one and "
                "the mesh built for the other"
            )
        if parallel.cfg_parallel_enabled:
            raise NotImplementedError(
                "Qwen-Image is guidance-distilled (one forward per step); CFG parallelism does "
                "not apply"
            )
        # The mesh carries no SP flag (MeshSpec drops it), so the module is the only place SP
        # takes effect: a config without it would build the dense model under an SP launch,
        # and one with it would shard where the caller asked for none.
        if bool(config.sp_enabled) != bool(getattr(parallel, "sp_enabled", False)):
            raise ValueError(
                f"NeuronQwenImageConfig has sp_enabled={config.sp_enabled} but the parallel "
                f"config has sp_enabled={getattr(parallel, 'sp_enabled', False)}"
            )
        cp = int(parallel.cp_degree)
        if bool(config.context_parallel_enabled) != (cp > 1):
            raise ValueError(
                f"NeuronQwenImageConfig has context_parallel_enabled="
                f"{config.context_parallel_enabled} but the parallel config has cp_degree={cp}"
            )
        if cp > 1 and str(config.cp_mode) != str(parallel.cp_mode):
            raise ValueError(
                f"NeuronQwenImageConfig has cp_mode={config.cp_mode!r} but the parallel config "
                f"has cp_mode={parallel.cp_mode!r}"
            )
        if config.cp_degree is None:
            config = dataclasses.replace(config, cp_degree=cp)
        elif int(config.cp_degree) != cp:
            raise ValueError(
                f"NeuronQwenImageConfig has cp_degree={config.cp_degree} but the parallel config "
                f"has cp_degree={cp}"
            )
        config.validate()
        if self.device.type == "neuron" and self.dtype is not torch.bfloat16:
            raise ValueError(
                f"the Qwen-Image DiT runs in torch.bfloat16 on the neuron device, got "
                f"{self.dtype} (decision D42: the TP layers all-reduce in the activation dtype, "
                "checked against NxD for bf16 only)"
            )
        self.config = config
        # Blocks past num_layers are in the checkpoint on purpose when num_layers was cut.
        total = config.checkpoint_num_layers
        if total is not None and int(total) > int(config.num_layers):
            self.skipped_checkpoint_prefixes = tuple(
                f"transformer_blocks.{i}." for i in range(int(config.num_layers), int(total))
            )

    def build_module(self) -> torch.nn.Module:
        return _trace_module_class()(self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The DiT on ``(hidden_states, timestep, encoder_hidden_states)``; the orchestrator's
        mask and guidance are accepted and dropped here, before the base's shape bookkeeping.

        The module would drop them anyway (no mask, see the module docstring; no guidance
        embedder, ``NeuronQwenImageConfig.validate``), but passed through, the mask tensor would
        make a compile-mode call's input shapes differ from the warm-up's three, and the base
        would warn of (and record) an uncompiled shape for a call whose blocks hit the warm-up
        graph.
        """
        del encoder_hidden_states_mask, guidance
        return super().forward(hidden_states, timestep, encoder_hidden_states)

    def checkpoint_key(self, name: str) -> str:
        return checkpoint_key(name)

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """``(hidden_states, timestep, encoder_hidden_states)`` at the static shape, in ``dtype``.

        The packed latents ``[B, image_seq_len, in_channels]``, the per-sample timestep and the
        zero-padded text embeddings ``[B, text_seq_len, joint_attention_dim]``: exactly the
        per-step shapes the orchestrator sends (one forward per step, no CFG batch), so the
        warm-up compiles the only block graph the run needs. Under CP every rank receives the
        full sequence and the module scatters it.
        """
        cfg = self.config
        batch = int(cfg.batch_size)
        return (
            torch.zeros(batch, cfg.image_seq_len, int(cfg.in_channels), dtype=self.dtype),
            torch.zeros(batch, dtype=self.dtype),
            torch.zeros(
                batch, int(cfg.text_seq_len), int(cfg.joint_attention_dim), dtype=self.dtype
            ),
        )


__all__ = ["CHECKPOINT_PREFIX", "NeuronQwenImageTransformerApplication", "checkpoint_key"]
