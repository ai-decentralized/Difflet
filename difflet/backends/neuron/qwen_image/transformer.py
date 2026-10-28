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

The text mask is applied, as diffusers applies it: prompts are zero-padded to ``text_seq_len``
(the one static shape) and the mask's valid-row count bounds the attended keys. The trace module
turns the mask into that count (``text_valid_len``) and the backend-neutral attention processor
(``_QwenImageTrainiumAttnProcessor._masked_joint_attention``) orders the keys ``[image ‖ text]``
so the valid ones are the prefix ``[0, image tokens + count)``, which the key-bounded flash
kernel takes (ring and Ulysses CP mask by the same count). The Trainium and TPU closures still
drop the mask and attend the padding: a zero row is not an absent key (``txt_norm``/``txt_in``
turn it into ``txt_in``'s bias), and on the 2026-10-26 fox prompt (17 valid rows of 1024) attending
them moved the CPU fp32 noise prediction by 6.5 % (rel. L2) at t=1, against 2.7 % for bf16.
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

#: The dtype the text mask reaches the module in, the warm-up's (HunyuanVideo's int64 mask
#: precedent); any other mask dtype is cast to it first.
_MASK_DTYPE = torch.int64
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
    """``_QwenImageTransformerTraceModule`` with the text mask applied, the mask and guidance
    optional, and CFG parallelism.

    The mask (``[B, text_seq_len]``, a prefix of valid rows) becomes the valid-row count
    ``text_valid_len`` the base forward hands the attention processor; without one the text is
    attended unmasked (the base's behaviour). The warm-up broadcasts tensors only
    (``broadcast_tensor`` has no None), so it calls the module with the four tensors the DiT
    consumes; the orchestrator's five-argument call keeps working. Built lazily: importing the
    modeling binds ``difflet.ops`` to the selected backend, which must not happen when this
    module is merely imported.

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
            text_valid_len = None
            if encoder_hidden_states_mask is not None:
                text_valid_len = encoder_hidden_states_mask.to(torch.int32).sum(dim=1)
            if not self.cfg_parallel_enabled:
                return super().forward(
                    hidden_states, timestep, encoder_hidden_states, None, guidance,
                    text_valid_len=text_valid_len,
                )
            rank = get_cfg_rank_spmd(self.cfg_rank_util.get_rank())

            def mine(tensor):
                return scatter_to_process_group_spmd(
                    tensor, partition_dim=0, rank=rank, process_group=self.cfg_group
                )

            output = super().forward(
                mine(hidden_states), mine(timestep), mine(encoder_hidden_states), None,
                None if guidance is None else mine(guidance),
                text_valid_len=None if text_valid_len is None else mine(text_valid_len),
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
        """The DiT on ``(hidden_states, timestep, encoder_hidden_states,
        encoder_hidden_states_mask)``; guidance is accepted and dropped here (no guidance
        embedder, ``NeuronQwenImageConfig.validate``).

        The mask (``[B, text_seq_len]``, True / 1 on the prompt's valid rows, a prefix: the
        embeddings are right-padded) reaches the module as int64, the dtype the warm-up compiled
        for, whatever the caller passed; the module counts its valid rows and the attention
        masks the rest of the text keys (see the module docstring). Without one every text row
        is valid: an all-ones mask is made at the input's shape, so a three-tensor call hits the
        warm-up's graph too.
        """
        del guidance
        if encoder_hidden_states_mask is None:
            encoder_hidden_states_mask = torch.ones(
                encoder_hidden_states.shape[:2], dtype=_MASK_DTYPE,
                device=encoder_hidden_states.device,
            )
        elif encoder_hidden_states_mask.dtype != _MASK_DTYPE:
            encoder_hidden_states_mask = encoder_hidden_states_mask.to(_MASK_DTYPE)
        return super().forward(
            hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask
        )

    def checkpoint_key(self, name: str) -> str:
        return checkpoint_key(name)

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """``(hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask)`` at the
        static shape, the floats in ``dtype``, the mask int64 and all ones.

        The packed latents ``[B, image_seq_len, in_channels]``, the per-sample timestep, the
        zero-padded text embeddings ``[B, text_seq_len, joint_attention_dim]`` and their mask
        ``[B, text_seq_len]``: exactly the per-call shapes the orchestrator sends, so the
        warm-up compiles the only block graph the run needs (the valid-row count is a tensor,
        so no prompt length compiles anew). B is 1, or 2 under CFG parallelism (the
        ``[negative, conditional]`` pair, which the module splits over the CFG group: each
        rank's blocks still see batch 1). Under CP every rank receives the full sequence and
        the module scatters it.
        """
        cfg = self.config
        batch = cfg.dit_batch
        return (
            torch.zeros(batch, cfg.image_seq_len, int(cfg.in_channels), dtype=self.dtype),
            torch.zeros(batch, dtype=self.dtype),
            torch.zeros(
                batch, int(cfg.text_seq_len), int(cfg.joint_attention_dim), dtype=self.dtype
            ),
            torch.ones(batch, int(cfg.text_seq_len), dtype=_MASK_DTYPE),
        )


__all__ = ["CHECKPOINT_PREFIX", "NeuronQwenImageTransformerApplication", "checkpoint_key"]
