"""Neuron application for the HunyuanVideo 1.0 DiT.

The module is Difflet's backend-neutral ``HunyuanVideoTransformer3DModel`` (``difflet/models/
hunyuan_video/modeling_hunyuan_video.py``), the class the Trainium path compiles and the one the
CPU parity test holds to diffusers (``tests/numerical/test_hunyuan_video_vs_diffusers.py``); HV1.5's
neuron port wraps diffusers' class only because no Difflet port of 1.5 exists. The port builds
its own tensor parallelism from ``difflet.ops`` (q/k/v and the MLP inputs column-parallel, the
attention and MLP outputs row-parallel, 24 heads split over tp; the port adds the blocks' AdaLN
modulation linears, column-parallel with the output gathered), its context parallelism (image
tokens scattered over the CP group, the text replicated) and its sequence parallelism (the image
stream of the dual-stream blocks sharded over the TP group), and ``difflet.ops`` dispatches to
this backend's ``ops_impl``. Here the lifecycle is ``TorchNeuronApplicationBase``'s: the module
is built on ``meta`` after the mesh exists (so the TP layers size their shards and the CP
attention captures its group), each rank loads its own slices of the diffusers checkpoint in
``dtype``, and in compile mode the 20 dual-stream ``transformer_blocks`` and the 40 single-stream
``single_transformer_blocks`` are compiled in place into one graph per block type while the top
level (patch embedding, token refiner, condition embedding, rotary tables, the joint mask, final
norm and projection, the CP scatter and gather) stays eager. The warm-up runs at the one static
shape of ``NeuronHunyuanVideoConfig``.

The module is a subclass whose ``forward`` takes the six tensors positionally and returns the
noise prediction as a tensor: the base warms up and compiles with a tuple of broadcast tensors,
which has no room for ``return_dict``.

Checkpoint mapping: as in Flux, diffusers fuses each single-stream block's attention and MLP
output projections into one ``proj_out`` (input columns ``[0, inner)`` attention, ``[inner, 5
inner)`` MLP), and the port splits them so each is row-parallel on its own input sharding.
``proj_out_attn.weight`` and ``proj_out_mlp.weight`` are therefore column ranges of
``proj_out.weight`` (read per TP shard), and ``proj_out_attn.bias`` is ``proj_out.bias``
(``proj_out_mlp`` has none): ``core/checkpoint.py``'s ``single_block_proj_out_*``, shared with
Flux. Every other key is diffusers' own. A config cut below the checkpoint's depth
(``num_layers=N``, ``num_single_layers=M``) loads the first blocks of each stack and reports the
others under ``load_report["skipped"]``.

The text padding is a key count, not a dense mask: the port's joint attention turns the
right-padded text mask into per-query key bounds (image keys plus the valid text keys), which
on the device run attention_cte's key-bounded flash kernel. Numerics depend on two backend
workarounds installed by ``prepare_runtime`` (``core/workarounds.py``): fp32 LayerNorm for bf16
inputs (every adaptive norm is a LayerNorm) and the bool bitwise guard (the token refiner builds
its mask as ``m & m.transpose(2, 3)``).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.backends.neuron.core.checkpoint import (
    CheckpointSlice,
    single_block_proj_out_key,
    single_block_proj_out_slices,
)
from difflet.backends.neuron.core.config_checks import (
    check_flag_matches,
    check_tp_matches,
    require_bf16_on_device,
    resolve_cp_degree,
    truncated_block_prefixes,
)
from difflet.backends.neuron.hunyuan_video.config import NeuronHunyuanVideoConfig

#: Valid text tokens in the warm-up's mask: any count compiles the same block graphs (the
#: mask is a key bound, not a shape).
_EXAMPLE_VALID_TEXT = 64
#: The warm-up's guidance: scale 6.0 x 1000, as the pipeline sends it.
_EXAMPLE_GUIDANCE = 6000.0


def checkpoint_key(name: str) -> str:
    """Diffusers checkpoint key holding the module parameter ``name``."""
    return single_block_proj_out_key(name)


def _tensor_signature_class():
    """``HunyuanVideoTransformer3DModel`` with a tensors-only call: six positional tensors in,
    the noise prediction out.

    Built lazily: importing the modeling binds ``difflet.ops`` to the selected backend, which
    must not happen when this module is merely imported.
    """
    from difflet.models.hunyuan_video.modeling_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    class _HunyuanVideoTensorSignature(HunyuanVideoTransformer3DModel):
        def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
            pooled_projections: torch.Tensor,
            guidance: torch.Tensor,
        ) -> torch.Tensor:
            return super().forward(
                hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                pooled_projections, guidance, return_dict=False,
            )[0]

    return _HunyuanVideoTensorSignature


def _shard_modulation_linears(module: torch.nn.Module) -> None:
    """Make the blocks' AdaLN modulation linears column-parallel with the output gathered:
    each dual-stream block's ``norm1.linear`` and ``norm1_context.linear`` (3072 -> 6 x 3072)
    and each single-stream block's ``norm.linear`` (3072 -> 3 x 3072).

    The shared modeling builds them as diffusers' ``AdaLayerNormZero`` /
    ``AdaLayerNormZeroSingle``, whose ``linear`` is a plain ``nn.Linear``, so every rank would
    hold all 80 of them (6.80 GB in bf16). Their input is the per-sample ``[B, inner]``
    conditioning vector, so gathering the output is one tiny all-gather per layer; the loader
    reads each rank's rows of the unchanged checkpoint keys (``_difflet_shard``). The swap is
    here and not in the shared modeling, which the Trainium path also compiles. HV1.5's port
    does the same (``hunyuan_video15/transformer.py``), as does this model's own token
    refiner."""
    from difflet.backends.neuron.ops_impl.linear import ColumnParallelLinear

    def col(lin: torch.nn.Linear) -> ColumnParallelLinear:
        return ColumnParallelLinear(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                    gather_output=True, dtype=lin.weight.dtype,
                                    device=lin.weight.device)

    for block in module.transformer_blocks:
        block.norm1.linear = col(block.norm1.linear)
        block.norm1_context.linear = col(block.norm1_context.linear)
    for block in module.single_transformer_blocks:
        block.norm.linear = col(block.norm.linear)


class NeuronHunyuanVideoTransformerApplication(TorchNeuronApplicationBase):
    """The HunyuanVideo DiT, sharded across the TP group, at the config's static shape.

    Without ``config`` it is read from ``model_path`` (the ``transformer`` directory) with the
    topology of ``parallel`` and ``overrides`` for the rest (``height=``, ``num_frames=``, ...);
    with one, ``config.tp_degree``, ``sp_enabled`` and ``cfg_parallel_enabled`` must equal the
    parallel config's (the module is sized by the config, the mesh by ``parallel``) and
    ``cp_degree`` is filled from it. On the neuron device the dtype must be ``torch.bfloat16``
    (decision D42). A config cut below the checkpoint's depth loads the first blocks of each
    stack and reports the others under ``load_report["skipped"]``.
    """

    block_attrs = ("transformer_blocks", "single_transformer_blocks")

    def __init__(
        self,
        *,
        model_path,
        config: NeuronHunyuanVideoConfig | None = None,
        parallel=None,
        dtype=torch.bfloat16,
        exec_mode: str | None = None,
        device="neuron",
        **overrides,
    ) -> None:
        super().__init__(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            exec_mode=exec_mode,
            device=device,
        )
        if config is None:
            p = self.parallel
            config = NeuronHunyuanVideoConfig.from_pretrained(
                model_path, tp_degree=int(p.tp_degree),
                context_parallel_enabled=int(p.cp_degree or 1) > 1, cp_mode=p.cp_mode,
                sp_enabled=bool(p.sp_enabled), cfg_parallel_enabled=bool(p.cfg_parallel_enabled),
                **overrides,
            )
        elif overrides:
            raise TypeError(
                f"{type(self).__name__} takes either config or config overrides, not both; "
                f"got config and {sorted(overrides)}"
            )
        name = "NeuronHunyuanVideoConfig"
        check_tp_matches(name, config, self.parallel)
        check_flag_matches(name, config, self.parallel, "cfg_parallel_enabled")
        # The mesh carries no SP flag (MeshSpec drops it), so the module is the only place SP
        # takes effect.
        check_flag_matches(name, config, self.parallel, "sp_enabled")
        # The port calls init_parallel_mesh(config) again under CP: a no-op only when the
        # config and the parallel config describe the same mesh.
        config = resolve_cp_degree(name, config, self.parallel)
        config.validate()
        require_bf16_on_device("HunyuanVideo", self.device.type, self.dtype)
        self.config = config
        # Blocks past num_layers / num_single_layers are in the checkpoint on purpose when the
        # depth was cut.
        single = SimpleNamespace(num_layers=config.num_single_layers,
                                 checkpoint_num_layers=config.checkpoint_num_single_layers)
        skipped = (truncated_block_prefixes(config, "transformer_blocks.")
                   + truncated_block_prefixes(single, "single_transformer_blocks."))
        if skipped:
            self.skipped_checkpoint_prefixes = skipped

    def build_module(self) -> torch.nn.Module:
        module = _tensor_signature_class()(self.config)
        _shard_modulation_linears(module)
        return module

    def checkpoint_key(self, name: str) -> str:
        return checkpoint_key(name)

    def checkpoint_slices(self) -> dict[str, CheckpointSlice]:
        inner = self.config.inner_dim
        return single_block_proj_out_slices(self.config.num_single_layers, inner,
                                            int(inner * float(self.config.mlp_ratio)))

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """``(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
        pooled_projections, guidance)`` at the static shape: the latents ``[B, C, F', H/8,
        W/8]``, the per-sample timestep, the right-padded Llama embeddings ``[B, text_seq_len,
        text_embed_dim]`` with an int64 mask, the CLIP pooled projection and the guidance
        (scale x 1000). Exactly the per-step shapes the orchestrator sends, so the warm-up
        compiles the only block graphs the run needs. Under CP or SP every rank receives the
        whole latent and the module scatters it.
        """
        cfg, dt = self.config, self.dtype
        batch, text = int(cfg.batch_size), int(cfg.text_seq_len)
        mask = torch.zeros(batch, text, dtype=torch.int64)
        mask[:, :min(_EXAMPLE_VALID_TEXT, text)] = 1
        return (
            torch.zeros(batch, int(cfg.in_channels), cfg.latent_frames, cfg.latent_height,
                        cfg.latent_width, dtype=dt),
            torch.ones(batch, dtype=dt),
            torch.zeros(batch, text, int(cfg.text_embed_dim), dtype=dt),
            mask,
            torch.zeros(batch, int(cfg.pooled_projection_dim), dtype=dt),
            torch.full((batch,), _EXAMPLE_GUIDANCE, dtype=dt),
        )

    @torch.no_grad()
    def teacache_mod_input(self, hidden_states: torch.Tensor, timestep: torch.Tensor,
                           pooled_projections: torch.Tensor,
                           guidance: torch.Tensor) -> torch.Tensor:
        """The TeaCache signal, as the model defines it (``HunyuanVideoTransformer3DModel.
        teacache_mod_input``, the method the Trainium probe runs): block 0's modulated image
        input. Its modulation depends on the timestep, the pooled projection and the guidance;
        the method ignores the text, so none is passed. The modulation linears gather their
        output, so every rank holds the result whole. Host inputs; moved to this rank's
        device in the model's dtype."""
        dev, dt = self.device, self.dtype
        return self.module.teacache_mod_input(
            hidden_states.to(dev, dt), timestep.to(dev, dt), None, None,
            pooled_projections.to(dev, dt), guidance.to(dev, dt),
        )


__all__ = ["NeuronHunyuanVideoTransformerApplication", "checkpoint_key"]
