"""Neuron application for the LTX-2 audio-video DiT.

The Trainium counterpart is ``NeuronLTX2TransformerApplication``
(``difflet/backends/trainium/ltx_2/transformer.py``); both wrap the backend-neutral
``_LTX2TransformerTraceModule`` (``difflet/models/ltx_2/modeling_ltx_2.py``), which nests
diffusers' ``LTX2VideoTransformer3DModel`` as ``self.transformer`` and, under tensor
parallelism, head-shards each block's six attentions (video and audio self-attention, the two
text cross-attentions, audio-to-video and video-to-audio) and both feed-forwards with the
``difflet.ops`` TP layers, installs the TP attention processor (q/k RMS norm over the full inner
dim through one all-reduce of the local sums of squares, attention through
``difflet.ops.attention``) and slices the four RoPE tables to this rank's heads once per
forward. Here the lifecycle is ``TorchNeuronApplicationBase``'s: the module is built on ``meta``
after the mesh exists, each rank loads its own slices of the diffusers checkpoint in ``dtype``,
and in compile mode the 48 identical ``LTX2VideoTransformerBlock``s are compiled in place into
one shared graph while the top level (patch projections, timestep and caption embeddings, the
RoPE, the output norms and projections) stays eager.

Two neuron-specific changes, both outside the blocks:

* RoPE on the host. Diffusers computes the four rotary tables from the coordinates in float64
  (``rope_double_precision``), which the neuron device does not run. They are a function of the
  coordinates alone (no weights), so ``_HostRope`` computes them on the host CPU from the
  coordinates the call carries (with the TP head slice and the bf16 cast the TP patch adds),
  moves them to the device and keeps the last result: every denoise step sends the same
  coordinates, so they are computed once per process.
* No text mask. The connectors replace the prompt padding with learned registers, and the
  Trainium processor already attends the text unmasked (its masked branch runs the same
  unmasked kernel); this module is called without the masks (they are bool tensors, which the
  warm-up broadcast cannot carry). Every Difflet LTX-2 backend therefore has the same semantics.

The call is ``(hidden_states, audio_hidden_states, encoder_hidden_states,
audio_encoder_hidden_states, timestep, video_coords, audio_coords)``; ``sigma`` (LTX-2.3 prompt
modulation only) is the timestep, as the orchestrator passes it.
"""

from __future__ import annotations

import dataclasses

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.backends.neuron.ltx_2.config import NeuronLtx2Config

#: The trace module nests the diffusers model under this attribute; checkpoint keys are the
#: diffusers model's own (``transformer_blocks.0.attn1.to_q.weight``, ``proj_out.bias``).
CHECKPOINT_PREFIX = "transformer."
#: The diffusers model's rotary-embedding modules, each computed once per forward.
ROPE_ATTRS = ("rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope")


def checkpoint_key(name: str) -> str:
    """Diffusers checkpoint key holding the module parameter ``name``: the TP replacements keep
    diffusers' attribute paths (``attn1.to_q``, ``attn1.to_out[0]``, ``ff.net[0].proj``,
    ``ff.net[2]``), so only the wrapper prefix differs."""
    return name.removeprefix(CHECKPOINT_PREFIX)


class _HostRope(torch.nn.Module):
    """One diffusers ``LTX2AudioVideoRotaryPosEmbed``, evaluated on the host CPU.

    ``forward(coords, device)`` is the wrapped module's (``device`` is where diffusers wants the
    tables), computed from a CPU copy of ``coords`` (float64 frequencies as in diffusers) and
    moved to ``device``. The wrapped module's forward includes the TP patch
    (``_patch_ltx2_rope_for_tp``: this rank's heads, bf16) when tp > 1. The last coordinates and
    their device tables are kept, and an equal call returns them without recomputing. The
    wrapped module has no parameters or buffers, so the state dict is unchanged; it is held
    outside the module tree.
    """

    def __init__(self, rope: torch.nn.Module) -> None:
        super().__init__()
        self._rope = (rope,)
        self._last: tuple[torch.Tensor, torch.device, tuple[torch.Tensor, torch.Tensor]] | None
        self._last = None
        self.computed = 0  # how many times the tables were computed (evidence for tests)

    @property
    def modality(self) -> str:
        return self._rope[0].modality

    def forward(self, coords: torch.Tensor, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.device(device) if device is not None else coords.device
        host = coords.detach().to("cpu")
        last = self._last
        if (last is not None and last[1] == target and last[0].shape == host.shape
                and torch.equal(last[0], host)):
            return last[2]
        with torch.no_grad():
            cos, sin = self._rope[0](host, device="cpu")
        tables = (cos.to(target), sin.to(target))
        self._last = (host.clone(), target, tables)
        self.computed += 1
        return tables


def _trace_module_class():
    """``_LTX2TransformerTraceModule`` with the host RoPE, no masks, and CFG parallelism.

    Built lazily: importing the modeling binds ``difflet.ops`` to the selected backend, which
    must not happen when this module is merely imported.

    With ``cfg_parallel_enabled`` the call carries the CFG pair ``[unconditional, conditional]``
    as batch 2 (the orchestrator's ``torch.cat([latents] * 2)`` with ``[negative, positive]``
    embeddings): each rank keeps the half of its CFG coordinate
    (``scatter_to_process_group_spmd`` over the CFG group, a local narrow), runs the batch-1 DiT
    on it in its own TP group, and the halves are all-gathered back along the batch, so every
    rank returns both predictions in order. The blocks still see batch 1, so the block graph is
    the non-CFG one. The base module's own CFG path (which also scatters the masks) is not used:
    the module is built with CFG off and splits the call here, as the Qwen-Image neuron module
    does.
    """
    from difflet.models.ltx_2.modeling_ltx_2 import _LTX2TransformerTraceModule
    from difflet.ops import (
        SPMDRank,
        gather_from_tensor_model_parallel_region_with_dim,
        get_cfg_group,
        get_cfg_rank_spmd,
        get_world_group,
        scatter_to_process_group_spmd,
    )

    class NeuronLtx2TraceModule(_LTX2TransformerTraceModule):
        def __init__(self, config: NeuronLtx2Config):
            super().__init__(dataclasses.replace(config, cfg_parallel_enabled=False))
            for attr in ROPE_ATTRS:
                setattr(self.transformer, attr, _HostRope(getattr(self.transformer, attr)))
            # The base forward's own CFG path stays off (its cfg_parallel_enabled is False);
            # the split happens here.
            self.cfg_split = bool(config.cfg_parallel_enabled)
            if self.cfg_split:
                self.cfg_group = get_cfg_group()
                self.cfg_rank_util = SPMDRank(world_size=get_world_group().size())

        def forward(
            self,
            hidden_states: torch.Tensor,
            audio_hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            audio_encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            video_coords: torch.Tensor,
            audio_coords: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            inputs = (hidden_states, audio_hidden_states, encoder_hidden_states,
                      audio_encoder_hidden_states, timestep, video_coords, audio_coords)
            if self.cfg_split:
                rank = get_cfg_rank_spmd(self.cfg_rank_util.get_rank())
                inputs = tuple(
                    scatter_to_process_group_spmd(
                        t, partition_dim=0, rank=rank, process_group=self.cfg_group
                    )
                    for t in inputs
                )
            hidden, audio, text, audio_text, t, vcoords, acoords = inputs
            video_out, audio_out = super().forward(
                hidden, audio, text, audio_text, t, t, None, None, vcoords, acoords
            )
            if self.cfg_split:
                video_out = gather_from_tensor_model_parallel_region_with_dim(
                    video_out, gather_dim=0, process_group=self.cfg_group
                )
                audio_out = gather_from_tensor_model_parallel_region_with_dim(
                    audio_out, gather_dim=0, process_group=self.cfg_group
                )
            return video_out, audio_out

    return NeuronLtx2TraceModule


def example_coords(config: NeuronLtx2Config, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The pipeline's RoPE coordinates at the config's static shape (``make_ltx_2_video_coords``
    and ``make_ltx_2_audio_coords`` with the checkpoint's scale factors), fp32."""
    from difflet.models.ltx_2.pipeline import make_ltx_2_audio_coords, make_ltx_2_video_coords

    video = make_ltx_2_video_coords(
        batch_size=batch, num_frames=config.latent_num_frames, height=config.latent_height,
        width=config.latent_width, device="cpu", patch_size=int(config.patch_size),
        patch_size_t=int(config.patch_size_t), scale_factors=tuple(config.vae_scale_factors),
        causal_offset=int(config.causal_offset), fps=float(config.frame_rate),
    )
    audio = make_ltx_2_audio_coords(
        batch_size=batch, audio_num_frames=config.audio_seq_len, device="cpu",
        patch_size_t=int(config.audio_patch_size_t), scale_factor=int(config.audio_scale_factor),
        causal_offset=int(config.causal_offset), sampling_rate=int(config.audio_sampling_rate),
        hop_length=int(config.audio_hop_length),
    )
    return video.contiguous(), audio.contiguous()


class NeuronLtx2TransformerApplication(TorchNeuronApplicationBase):
    """The LTX-2 DiT, sharded across the TP group, at the config's static shape.

    ``model_path`` is the transformer directory (``<model>/transformer``). The config sizes the
    module and the parallel config the mesh, so they must agree: ``tp_degree`` and
    ``cfg_parallel_enabled`` equal, no CP, SP or DP lanes (``NeuronLtx2Config.validate``).

    On the neuron device the dtype must be ``torch.bfloat16`` (decision D42, as for Wan and
    Qwen-Image: the neuron TP linears all-reduce in the activation dtype, checked against NxD for
    bf16 only). A ``num_layers`` below the checkpoint's (``NeuronLtx2Config.from_pretrained(...,
    num_layers=N)``) loads the first N blocks and reports the others under
    ``load_report["skipped"]``.
    """

    block_attrs = ("transformer.transformer_blocks",)

    def __init__(
        self,
        *,
        model_path,
        config: NeuronLtx2Config,
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
                f"NeuronLtx2Config has tp_degree={config.tp_degree} but the parallel config has "
                f"tp_degree={parallel.tp_degree}; the module would be sharded for one and the "
                "mesh built for the other"
            )
        if bool(config.cfg_parallel_enabled) != bool(parallel.cfg_parallel_enabled):
            raise ValueError(
                f"NeuronLtx2Config has cfg_parallel_enabled={config.cfg_parallel_enabled} but "
                f"the parallel config has cfg_parallel_enabled={parallel.cfg_parallel_enabled}"
            )
        for what, value in (("cp_degree", int(parallel.cp_degree)),
                            ("dp_degree", int(getattr(parallel, "dp_degree", 1) or 1))):
            if value > 1:
                raise NotImplementedError(
                    f"the LTX-2 DiT on the neuron backend runs tensor and CFG parallelism only; "
                    f"got {what}={value}"
                )
        if bool(getattr(parallel, "sp_enabled", False)):
            raise NotImplementedError(
                "the LTX-2 DiT on the neuron backend has no sequence parallelism (sp_enabled)"
            )
        config.validate()
        if self.device.type == "neuron" and self.dtype is not torch.bfloat16:
            raise ValueError(
                f"the LTX-2 DiT runs in torch.bfloat16 on the neuron device, got {self.dtype} "
                "(decision D42: the TP layers all-reduce in the activation dtype, checked "
                "against NxD for bf16 only)"
            )
        self.config = config
        total = config.checkpoint_num_layers
        if total is not None and int(total) > int(config.num_layers):
            self.skipped_checkpoint_prefixes = tuple(
                f"transformer_blocks.{i}." for i in range(int(config.num_layers), int(total))
            )

    def build_module(self) -> torch.nn.Module:
        return _trace_module_class()(self.config)

    def checkpoint_key(self, name: str) -> str:
        return checkpoint_key(name)

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """The seven call tensors at the static shape: the packed video and audio latents and
        the two text streams in ``dtype``, a mid-schedule timestep, and the pipeline's own fp32
        RoPE coordinates (so the warm-up also fills the host RoPE cache). B is 1, or 2 under
        CFG parallelism (the pair the module splits: each rank's blocks still see batch 1)."""
        cfg = self.config
        batch = cfg.dit_batch
        gen = torch.Generator().manual_seed(0)
        video_coords, audio_coords = example_coords(cfg, batch)
        return (
            torch.randn(batch, cfg.video_seq_len, int(cfg.in_channels), generator=gen
                        ).to(self.dtype),
            torch.randn(batch, cfg.audio_seq_len, int(cfg.audio_in_channels), generator=gen
                        ).to(self.dtype),
            torch.randn(batch, int(cfg.text_seq_len), cfg.video_text_dim, generator=gen
                        ).to(self.dtype),
            torch.randn(batch, int(cfg.text_seq_len), cfg.audio_text_dim, generator=gen
                        ).to(self.dtype),
            torch.full((batch,), 500.0, dtype=self.dtype),
            video_coords,
            audio_coords,
        )


__all__ = [
    "CHECKPOINT_PREFIX",
    "NeuronLtx2TransformerApplication",
    "ROPE_ATTRS",
    "checkpoint_key",
    "example_coords",
]
