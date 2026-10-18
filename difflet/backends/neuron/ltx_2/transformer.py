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

Context parallelism (``context_parallel_enabled``) shards the VIDEO tokens over the CP group and
keeps the short audio stream (51 frames at 480x704x49) and the text replicated, the layout Qwen-
Image's joint attention uses for its text: the module scatters the packed video latents and
their RoPE coordinates, every rank runs its video shard and the whole audio stream through the
blocks, and the video output is all-gathered back. Per block, only the two attentions whose
keys are video tokens need the other ranks' tokens: the video self-attention (``attn1``: all
three of q/k/v sharded; ``cp_mode`` gather_kv all-gathers k/v over the CP group, ulysses trades
the sequence shard for a head shard and back with two all-to-alls) and the video-to-audio
cross-attention (``video_to_audio_attn``: the replicated audio queries attend the gathered video
k/v). The text cross-attentions and the audio-to-video attention (video queries, audio keys) are
local. ``_Ltx2CPAttnProcessor`` is the TP processor with that k/v exchange added after the RoPE.
Ring attention is not wired: its partial-softmax merge would need the flash kernel's logsumexp
path for every block's self-attention, and gather_kv already moves only k/v (2310 video tokens x
4096 at 480x704x49).
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


#: Why the feed-forward GELU runs in fp32 on this backend. Measured on trn2 (one NeuronCore,
#: eager, 2310 x 4096 inputs of std 3): the device's bf16 ``gelu(approximate="tanh")`` is 1.0e-3
#: rel_l1 from fp32 against the CPU's bf16 6.9e-4. In the 4-block LTX-2 preflight at tp4 that
#: put the video output at t=500 at 1.54x the diffusers bf16 control's rel_l1 (gate: 1.15x);
#: with the activation in fp32 it is 1.05x, and every stream rounds like upstream bf16
#: (err_cos 0.66-0.74). The cast costs one fp32 elementwise pass over the FFN's local
#: intermediate per block. The Trainium path is unchanged.
_FP32_GELU = True


def _trace_module_class():
    """``_LTX2TransformerTraceModule`` with the host RoPE, no masks, CFG parallelism and the
    feed-forward GELU in fp32 (``_FP32_GELU``).

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
    from diffusers.models.activations import GELU

    from difflet.models.ltx_2.modeling_ltx_2 import _LTX2TransformerTraceModule
    from difflet.ops import (
        SPMDRank,
        gather_from_tensor_model_parallel_region_with_dim,
        get_cfg_group,
        get_cfg_rank_spmd,
        get_cp_group,
        get_cp_rank_spmd,
        get_world_group,
        scatter_to_process_group_spmd,
    )

    class Fp32GELU(GELU):
        """diffusers' ``GELU`` (the feed-forward's projection + tanh GELU) with the activation
        evaluated in fp32 and cast back: the neuron device's bf16 tanh GELU is less accurate
        than the CPU's (see ``_FP32_GELU``)."""

        def gelu(self, gate: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(gate.float(), approximate=self.approximate).to(
                gate.dtype)

    class NeuronLtx2TraceModule(_LTX2TransformerTraceModule):
        def __init__(self, config: NeuronLtx2Config):
            super().__init__(dataclasses.replace(config, cfg_parallel_enabled=False))
            for attr in ROPE_ATTRS:
                setattr(self.transformer, attr, _HostRope(getattr(self.transformer, attr)))
            for block in self.transformer.transformer_blocks:
                for ff in (block.ff, block.audio_ff):
                    act = ff.net[0]
                    if _FP32_GELU and type(act) is GELU:
                        act.__class__ = Fp32GELU  # same attributes and state dict keys
            # The base forward's own CFG path stays off (its cfg_parallel_enabled is False);
            # the split happens here.
            self.cfg_split = bool(config.cfg_parallel_enabled)
            if self.cfg_split:
                self.cfg_group = get_cfg_group()
                self.cfg_rank_util = SPMDRank(world_size=get_world_group().size())
            self.cp_split = bool(config.context_parallel_enabled)
            if self.cp_split:
                self._install_cp_processors(config)

        def _install_cp_processors(self, config: NeuronLtx2Config) -> None:
            """One processor per role, shared by every block (one compiled graph): the video
            self-attention exchanges k/v by ``cp_mode``, the video-to-audio attention gathers its
            video k/v, the four others attend locally."""
            cls = _cp_processor_class()
            tp = int(config.tp_degree)
            rank_util = getattr(self, "tp_rank_util", None) or SPMDRank(tp)
            self.cp_group = get_cp_group()
            self.cp_rank_util = SPMDRank(world_size=get_world_group().size())
            mode = "ulysses" if config.cp_mode == "ulysses" else "gather"
            roles = {"attn1": mode, "video_to_audio_attn": "gather"}
            processors = {
                kv: cls(tp_degree=tp, rank_util=rank_util, kv_mode=kv)
                for kv in {None, "gather", mode}
            }
            for block in self.transformer.transformer_blocks:
                for name in ("attn1", "audio_attn1", "attn2", "audio_attn2",
                             "audio_to_video_attn", "video_to_audio_attn"):
                    getattr(block, name).processor = processors[roles.get(name)]

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
            if self.cp_split:
                # This rank's video tokens and their RoPE coordinates (a local narrow).
                cp_rank = get_cp_rank_spmd(self.cp_rank_util.get_rank())
                hidden = scatter_to_process_group_spmd(
                    hidden, partition_dim=1, rank=cp_rank, process_group=self.cp_group)
                vcoords = scatter_to_process_group_spmd(
                    vcoords, partition_dim=2, rank=cp_rank, process_group=self.cp_group)
            video_out, audio_out = super().forward(
                hidden, audio, text, audio_text, t, t, None, None, vcoords, acoords
            )
            if self.cp_split:
                video_out = gather_from_tensor_model_parallel_region_with_dim(
                    video_out, gather_dim=1, process_group=self.cp_group
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


def _cp_processor_class():
    """``_LTX2TPAttnProcessor`` with the video tokens sharded over the CP group: after q/k
    normalisation and RoPE, ``kv_mode`` "gather" all-gathers k and v over the CP group (the
    queries stay local), "ulysses" runs the attention through ``difflet.ops.ulysses_attention``
    (all three sharded), None attends locally. Masks are ignored, as by the TP processor (the
    attention there is unmasked in both branches). Built lazily, like the trace module."""
    import math

    import diffusers.models.transformers.transformer_ltx2 as ltx2_transformer

    from difflet.models.ltx_2.modeling_ltx_2 import _LTX2TPAttnProcessor
    from difflet.ops import (
        attention,
        gather_from_tensor_model_parallel_region_with_dim,
        get_cp_group,
        ulysses_attention,
    )

    class Ltx2CPAttnProcessor(_LTX2TPAttnProcessor):
        def __init__(self, *, tp_degree: int, rank_util, kv_mode: str | None) -> None:
            super().__init__(tp_degree=tp_degree, rank_util=rank_util)
            if kv_mode not in (None, "gather", "ulysses"):
                raise ValueError(f"unknown kv_mode {kv_mode!r}")
            self.kv_mode = kv_mode
            self.cp_group = get_cp_group()

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, query_rotary_emb=None, key_rotary_emb=None):
            del attention_mask
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            query = self._global_rms_norm(attn.norm_q, query)
            key = self._global_rms_norm(attn.norm_k, key)
            if query_rotary_emb is not None:
                k_rope = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                if attn.rope_type == "interleaved":
                    query = ltx2_transformer.apply_interleaved_rotary_emb(query, query_rotary_emb)
                    key = ltx2_transformer.apply_interleaved_rotary_emb(key, k_rope)
                elif attn.rope_type == "split":
                    query = ltx2_transformer.apply_split_rotary_emb(query, query_rotary_emb)
                    key = ltx2_transformer.apply_split_rotary_emb(key, k_rope)
            out_dtype = query.dtype
            heads = attn.heads
            bsz, q_len, inner = query.shape
            head_dim = inner // heads

            def split(t):
                return t.reshape(bsz, t.shape[1], heads, head_dim).permute(0, 2, 1, 3)

            q4, k4, v4 = split(query), split(key), split(value)
            scale = 1.0 / math.sqrt(head_dim)
            if self.kv_mode == "ulysses":
                out = ulysses_attention(q4, k4, v4, scale=scale)
            else:
                if self.kv_mode == "gather":
                    k4 = gather_from_tensor_model_parallel_region_with_dim(
                        k4, gather_dim=2, process_group=self.cp_group)
                    v4 = gather_from_tensor_model_parallel_region_with_dim(
                        v4, gather_dim=2, process_group=self.cp_group)
                out = attention(q4, k4, v4, scale=scale, causal=False, tp_q=True, tp_k=True,
                                tp_out=False)
            hidden_states = out.permute(0, 2, 1, 3).reshape(bsz, q_len, inner).to(out_dtype)
            hidden_states = attn.to_out[0](hidden_states)
            return attn.to_out[1](hidden_states)

    return Ltx2CPAttnProcessor


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
        dp = int(getattr(parallel, "dp_degree", 1) or 1)
        if dp > 1:
            raise NotImplementedError(
                "the LTX-2 DiT on the neuron backend runs tensor, CFG and context parallelism; "
                f"got dp_degree={dp} (data parallelism runs as separate replicas)"
            )
        cp = int(parallel.cp_degree)
        if bool(config.context_parallel_enabled) != (cp > 1):
            raise ValueError(
                f"NeuronLtx2Config has context_parallel_enabled={config.context_parallel_enabled} "
                f"but the parallel config has cp_degree={cp}"
            )
        if cp > 1 and str(config.cp_mode) != str(parallel.cp_mode):
            raise ValueError(
                f"NeuronLtx2Config has cp_mode={config.cp_mode!r} but the parallel config has "
                f"cp_mode={parallel.cp_mode!r}"
            )
        if config.cp_degree is None:
            config = dataclasses.replace(config, cp_degree=cp)
        elif int(config.cp_degree) != cp:
            raise ValueError(
                f"NeuronLtx2Config has cp_degree={config.cp_degree} but the parallel config has "
                f"cp_degree={cp}"
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
