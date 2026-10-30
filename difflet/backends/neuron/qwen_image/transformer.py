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

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.backends.neuron.core.config_checks import (
    check_flag_matches,
    check_tp_matches,
    require_bf16_on_device,
    resolve_cp_degree,
    truncated_block_prefixes,
)
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
    """``_QwenImageTransformerTraceModule`` with the mask and guidance optional, and CFG
    parallelism.

    The warm-up broadcasts tensors only (``broadcast_tensor`` has no None), so it calls the
    module with the three tensors the DiT actually consumes; the orchestrator's five-argument
    call keeps working. Built lazily: importing the modeling binds ``difflet.ops`` to the
    selected backend, which must not happen when this module is merely imported.

    With ``cfg_parallel_enabled`` the call carries the true-CFG pair ``[negative,
    conditional]`` as batch 2 (``QwenImageOrchestrator._true_cfg_step``): each rank keeps the
    half of its CFG coordinate (``scatter_to_process_group_spmd`` over the CFG group, a local
    narrow: rank 0 of the group the negative branch, rank 1 the conditional one), runs the
    batch-1 DiT on it in its own TP group, and the halves are all-gathered back along the
    batch, so every rank returns both predictions in order, as Wan's model does. The blocks
    still see batch 1, so the block graph is the non-CFG one. This lives in the neuron
    subclass: the Trainium and TPU paths never run CFG parallelism for Qwen-Image.
    """
    from difflet.models.qwen_image.modeling_qwen_image import _QwenImageTransformerTraceModule
    from difflet.ops import (
        SPMDRank,
        gather_from_tensor_model_parallel_region_with_dim,
        get_cfg_group,
        get_cfg_rank_spmd,
        get_world_group,
        scatter_to_process_group_spmd,
    )

    class NeuronQwenImageTraceModule(_QwenImageTransformerTraceModule):
        def __init__(self, config):
            super().__init__(config)
            self.cfg_parallel_enabled = bool(getattr(config, "cfg_parallel_enabled", False))
            if self.cfg_parallel_enabled:
                self.cfg_group = get_cfg_group()
                self.cfg_rank_util = SPMDRank(world_size=get_world_group().size())

        def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_mask: torch.Tensor | None = None,
            guidance: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if not self.cfg_parallel_enabled:
                return super().forward(
                    hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask,
                    guidance,
                )
            rank = get_cfg_rank_spmd(self.cfg_rank_util.get_rank())

            def mine(tensor):
                return scatter_to_process_group_spmd(
                    tensor, partition_dim=0, rank=rank, process_group=self.cfg_group
                )

            output = super().forward(
                mine(hidden_states), mine(timestep), mine(encoder_hidden_states), None,
                None if guidance is None else mine(guidance),
            )
            return gather_from_tensor_model_parallel_region_with_dim(
                output, gather_dim=0, process_group=self.cfg_group
            )

    return NeuronQwenImageTraceModule


class NeuronQwenImageTransformerApplication(TorchNeuronApplicationBase):
    """The Qwen-Image DiT, sharded across the TP group (and the image tokens across the CP
    group), at the config's static shape.

    ``model_path`` is the transformer directory (``<model>/transformer``). The config sizes the
    module and the parallel config the mesh, so they must agree: ``tp_degree`` equal,
    ``context_parallel_enabled`` iff ``parallel.cp_degree > 1`` with the same ``cp_mode``,
    ``sp_enabled`` and ``cfg_parallel_enabled`` equal (CFG parallelism splits the batch-2
    true-CFG call over the CFG group; see ``_trace_module_class``). With CP on, the trace module calls
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
        name = "NeuronQwenImageConfig"
        check_tp_matches(name, config, self.parallel)
        check_flag_matches(name, config, self.parallel, "cfg_parallel_enabled")
        # The mesh carries no SP flag (MeshSpec drops it), so the module is the only place SP
        # takes effect: a config without it would build the dense model under an SP launch,
        # and one with it would shard where the caller asked for none.
        check_flag_matches(name, config, self.parallel, "sp_enabled")
        config = resolve_cp_degree(name, config, self.parallel)
        config.validate()
        require_bf16_on_device("Qwen-Image", self.device.type, self.dtype)
        self.config = config
        # Blocks past num_layers are in the checkpoint on purpose when num_layers was cut.
        skipped = truncated_block_prefixes(config, "transformer_blocks.")
        if skipped:
            self.skipped_checkpoint_prefixes = skipped

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
        per-call shapes the orchestrator sends, so the warm-up compiles the only block graph
        the run needs. B is 1, or 2 under CFG parallelism (the ``[negative, conditional]``
        pair, which the module splits over the CFG group: each rank's blocks still see batch
        1). Under CP every rank receives the full sequence and the module scatters it.
        """
        cfg = self.config
        batch = cfg.dit_batch
        return (
            torch.zeros(batch, cfg.image_seq_len, int(cfg.in_channels), dtype=self.dtype),
            torch.zeros(batch, dtype=self.dtype),
            torch.zeros(
                batch, int(cfg.text_seq_len), int(cfg.joint_attention_dim), dtype=self.dtype
            ),
        )


__all__ = ["CHECKPOINT_PREFIX", "NeuronQwenImageTransformerApplication", "checkpoint_key"]
