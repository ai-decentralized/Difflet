"""Backend-neutral modeling for the LTX-2 audio-video DiT transformer.

Extracted verbatim from ``difflet/backends/trainium/ltx_2/transformer.py`` so a second hardware
backend (the PyTorch-native neuron backend) can reuse it. Nothing here imports a backend: only
``torch``, pure-torch diffusers pieces and ``difflet.ops``, which dispatches to the selected
backend's implementation -- the shape the Wan and Qwen-Image modeling modules already have. The
Trainium file keeps the pieces that genuinely are Trainium (``LTX2TransformerInferenceConfig``,
the NxD ``ModelWrapper`` and ``NeuronApplicationBase``) and re-exports everything below, so every
existing importer keeps working unchanged.

This code patches an upstream diffusers ``LTX2VideoTransformer3DModel``: under tensor parallelism
it head-shards the six attention paths of each block and the two feed-forwards, installs a TP-aware
attention processor (global RMS over the sharded q/k, attention through ``difflet.ops.attention``)
and slices the RoPE tables to this rank's heads once per forward. ``config`` is duck-typed -- it
needs the LTX-2 geometry attributes (``in_channels``, ``latent_num_frames``, ``audio_num_frames``,
...), not any particular config class.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn

from difflet.common.tp_reduce import config_reduce_dtype
from difflet.ops.collectives import safe_tensor_parallel_size
from difflet.ops import (
    ColumnParallelLinear,
    RowParallelLinear,
    SPMDRank,
    attention as difflet_attention,
    gather_from_tensor_model_parallel_region_with_dim,
    get_cfg_group,
    get_cfg_rank_spmd,
    get_tensor_model_parallel_size,
    get_world_group,
    init_parallel_mesh,
    reduce_from_tensor_model_parallel_region,
    scatter_to_process_group_spmd,
)


def _difflet_apply_split_rotary_emb(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Equivalent split RoPE with explicit per-head reshape for XLA tracing."""

    cos, sin = freqs
    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        batch_size, num_heads, seq_len, _ = cos.shape
        head_dim = x.shape[-1] // num_heads
        x = x.reshape(batch_size, seq_len, num_heads, head_dim).swapaxes(1, 2)
        needs_reshape = True

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(f"Expected x.shape[-1] to be even for split rotary, got {last}.")
    rotary_dim = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, rotary_dim).float()
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]

    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]

    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    out = out.reshape(*out.shape[:-2], last)
    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(batch_size, seq_len, -1)
    return out.to(dtype=x_dtype)


# ── Tensor-parallel sharding (ports the validated AWS contrib recipe) ────────
# Reference: aws-neuron/neuronx-distributed-inference/contrib/models/ltx2-video-audio
# and /home/ubuntu/Armin-Neuron/ltx2/native-pytorch (validated TP=4, ~10 GB/rank).
#
# Per block (x48) the shardable linears are the six LTX2Attention paths plus the
# two FeedForwards. Q/K/V + FFN up-proj are column-parallel (gather_output=False);
# the attention output proj + FFN down-proj are row-parallel (input_is_parallel).
#
# Two LTX-2-specific correctness fixes vs a naive qwen-style swap:
#   * ``qk_norm="rms_norm_across_heads"`` normalizes q/k over the FULL inner dim
#     (all heads jointly, WITH an affine weight). Under head-sharding each rank
#     only holds inner_dim/tp features, so we all-reduce the local sum-of-squares
#     for the global RMS denominator and slice the affine weight to this rank.
#   * ``"split"`` RoPE returns cos/sin shaped [B, H, T, d/2] whose values differ
#     per head. The NxD graph is traced once at rank 0, so the head slice must use
#     the runtime rank (SPMDRank) — a Python int would bake rank 0 into all ranks.

_LTX2_ATTN_ATTRS = (
    "attn1",                # video self-attention
    "audio_attn1",          # audio self-attention
    "attn2",                # video <- text cross-attention
    "audio_attn2",          # audio <- text cross-attention
    "audio_to_video_attn",  # a2v cross-attention (Q: video, K/V: audio)
    "video_to_audio_attn",  # v2a cross-attention (Q: audio, K/V: video)
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def _safe_tensor_parallel_size() -> int:
    """tp_degree if a TP group is initialized (inside ModelBuilder), else 1.

    The host CPU copy in ``_load_cpu_transformer`` is built outside any parallel
    context and must stay unsharded (NxD asserts there). Narrow on purpose
    (``difflet.ops.collectives.safe_tensor_parallel_size``, shared with Qwen-Image):
    real config/import errors, and the neuron backend's uninitialised multi-rank
    mesh, surface instead of silently falling back to tp=1 and OOMing later.
    """
    return safe_tensor_parallel_size(get_tensor_model_parallel_size)


def _column_parallel_like(linear: nn.Linear, *, gather_output: bool) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        gather_output=gather_output,
    )


def _row_parallel_like(
    linear: nn.Linear, *, input_is_parallel: bool, reduce_dtype: torch.dtype
) -> RowParallelLinear:
    # reduce_dtype has no default on purpose: NxD's own default is fp32.
    return RowParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        input_is_parallel=input_is_parallel,
        reduce_dtype=reduce_dtype,
    )


class _LTX2TPAttnProcessor:
    """LTX-2 attention processor that stays correct under head tensor-parallelism.

    Faithful to the stock ``LTX2AudioVideoAttnProcessor`` except:
      * ``norm_q``/``norm_k`` use a global (all-reduced) RMS over the full inner
        dim and slice the replicated affine weight to this rank's heads.
      * The precomputed RoPE is sliced to this rank's heads via the runtime rank.
    The padding mask reshape uses the (already-sharded) ``attn.heads`` so it is
    sized to this rank's local head count automatically.
    """

    def __init__(self, *, tp_degree: int, rank_util: "SPMDRank") -> None:
        # RoPE is pre-sliced once at the rope modules (_patch_ltx2_rope_for_tp);
        # rank_util is still needed to slice the (replicated) qk-norm affine
        # weight, which the NxD weight loader does not shard for a plain RMSNorm.
        self.tp_degree = int(tp_degree)
        self._rank_util = rank_util

    def _global_rms_norm(self, norm, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        local_sq = x.float().pow(2).sum(dim=-1, keepdim=True)
        # Stays fp32 under --tp-reduce-dtype bf16: a [B, S, 1] RMS statistic.
        global_sq = reduce_from_tensor_model_parallel_region(local_sq)
        full_dim = in_dim * self.tp_degree
        eps = getattr(norm, "eps", None)
        eps = 1e-6 if eps is None else eps
        x_normed = x.float() * torch.rsqrt(global_sq / full_dim + eps)
        weight = getattr(norm, "weight", None)
        if weight is not None:
            x_normed = x_normed * _tp_head_scatter(weight, 0, self._rank_util).float()
        return x_normed.to(x.dtype)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
    ):
        import diffusers.models.transformers.transformer_ltx2 as ltx2_transformer

        # NOTE: gated attention is rejected in _shard_ltx2_transformer before this
        # processor is ever attached, so no guard is needed here.
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = self._global_rms_norm(attn.norm_q, query)
        key = self._global_rms_norm(attn.norm_k, key)

        if query_rotary_emb is not None:
            # RoPE was already sliced to this rank's heads + cast at the rope
            # modules (_patch_ltx2_rope_for_tp); apply directly.
            k_rope = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            if attn.rope_type == "interleaved":
                query = ltx2_transformer.apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = ltx2_transformer.apply_interleaved_rotary_emb(key, k_rope)
            elif attn.rope_type == "split":
                query = ltx2_transformer.apply_split_rotary_emb(query, query_rotary_emb)
                key = ltx2_transformer.apply_split_rotary_emb(key, k_rope)

        out_dtype = query.dtype
        n_heads = attn.heads
        if attention_mask is None:
            # Self-attention (the dominant cost): route to the NKI attention_cte
            # flash kernel in the [B*H, S, D] tp_q layout (~4x over compiled SDPA),
            # matching the other difflet video models. No mask, so no in-graph
            # mask->bounds resolution.
            bsz, q_len, inner = query.shape
            k_len = key.shape[1]
            head_dim = inner // n_heads
            q3 = query.reshape(bsz, q_len, n_heads, head_dim).permute(0, 2, 1, 3).reshape(
                bsz * n_heads, q_len, head_dim
            )
            k3 = key.reshape(bsz, k_len, n_heads, head_dim).permute(0, 2, 1, 3).reshape(
                bsz * n_heads, k_len, head_dim
            )
            v3 = value.reshape(bsz, k_len, n_heads, head_dim).permute(0, 2, 1, 3).reshape(
                bsz * n_heads, k_len, head_dim
            )
            out3 = difflet_attention(
                q3, k3, v3,
                scale=1.0 / math.sqrt(head_dim),
                causal=False,
                tp_q=True, tp_k=True, tp_out=False,
            )
            hidden_states = out3.reshape(bsz, n_heads, q_len, head_dim).permute(0, 2, 1, 3).reshape(
                bsz, q_len, inner
            )
        else:
            # Cross-attention (text key-padding mask): run UNMASKED through attention_cte
            # (the same NKI flash kernel as self-attn) rather than the slow SDPA fallback.
            # attention_cte's bound_min/bound_max (sequence-packing) path is self-attn only
            # (seqlen_q == seqlen_kv) and fails neuronx-cc for cross-attn (q_len != kv_len,
            # NCC_IBIR243), but the *unmasked* kernel supports q_len != kv_len (cf. wan
            # cross-attn). Dropping the mask attends over text padding; that is lossless
            # only if the text encoder's padding embeddings are benign (as UMT5's are for
            # wan) — gated by an explicit parity check vs the masked-SDPA baseline.
            bsz, qx_len, inner = query.shape
            kx_len = key.shape[1]
            head_dim = inner // n_heads
            q3 = query.reshape(bsz, qx_len, n_heads, head_dim).permute(0, 2, 1, 3).reshape(
                bsz * n_heads, qx_len, head_dim
            )
            k3 = key.reshape(bsz, kx_len, n_heads, head_dim).permute(0, 2, 1, 3).reshape(
                bsz * n_heads, kx_len, head_dim
            )
            v3 = value.reshape(bsz, kx_len, n_heads, head_dim).permute(0, 2, 1, 3).reshape(
                bsz * n_heads, kx_len, head_dim
            )
            out3 = difflet_attention(
                q3, k3, v3,
                scale=1.0 / math.sqrt(head_dim),
                causal=False,
                tp_q=True, tp_k=True, tp_out=False,
            )
            hidden_states = out3.reshape(bsz, n_heads, qx_len, head_dim).permute(0, 2, 1, 3).reshape(
                bsz, qx_len, inner
            )

        hidden_states = hidden_states.to(out_dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _tp_head_scatter(tensor: torch.Tensor, dim: int, rank_util: "SPMDRank") -> torch.Tensor:
    """This rank's contiguous chunk of ``tensor`` along ``dim`` (NxD SPMD scatter).

    Uses scatter_to_process_group_spmd (not narrow/index_select) for the Neuron
    compiler reasons documented elsewhere; non-zero dims are moved to dim 0 first
    since the primitive only exercises partition_dim=0.
    """
    rank = rank_util.get_rank()
    if dim == 0:
        return scatter_to_process_group_spmd(tensor, 0, rank, None)
    moved = tensor.movedim(dim, 0).contiguous()
    moved = scatter_to_process_group_spmd(moved, 0, rank, None)
    return moved.movedim(0, dim)


def _patch_ltx2_rope_for_tp(transformer: nn.Module, rank_util: "SPMDRank") -> None:
    """Slice each RoPE module's cos/sin to this rank's heads ONCE per forward.

    The 4 rope modules are each called once in ``transformer.forward`` and their
    outputs fan out to all 48 blocks, so slicing here (rather than inside every
    attention processor) removes hundreds of redundant per-rank scatters. Also
    casts to bf16 at the boundary (AWS contrib fix #5/#8).
    """
    for attr in ("rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope"):
        rope = getattr(transformer, attr, None)
        if rope is None:
            continue

        def _make(orig_forward):
            def _wrapped(*args, **kwargs):
                out = orig_forward(*args, **kwargs)
                if not (isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0])):
                    return out
                cos, sin = out
                if cos.ndim == 4:  # split RoPE [B, H, T, d/2] -> head axis
                    cos = _tp_head_scatter(cos, 1, rank_util)
                    sin = _tp_head_scatter(sin, 1, rank_util)
                elif cos.ndim == 3:  # interleaved RoPE [B, T, inner] -> last axis
                    cos = _tp_head_scatter(cos, -1, rank_util)
                    sin = _tp_head_scatter(sin, -1, rank_util)
                return cos.to(torch.bfloat16), sin.to(torch.bfloat16)

            return _wrapped

        rope.forward = _make(rope.forward)


def _shard_ltx2_transformer(
    transformer: nn.Module,
    tp_degree: int,
    rank_util: "SPMDRank",
    *,
    reduce_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Tensor-parallel shard LTX-2's attention + FFN linears across ``tp_degree`` ranks.

    ``reduce_dtype`` is the precision of the row-parallel all-reduce.
    """
    if tp_degree <= 1:
        return

    replicate_attn = _env_flag("DIFFLET_LTX2_TP_REPLICATE_ATTN")
    replicate_mlp = _env_flag("DIFFLET_LTX2_TP_REPLICATE_MLP")
    processor = _LTX2TPAttnProcessor(tp_degree=tp_degree, rank_util=rank_util)

    for block in transformer.transformer_blocks:
        if not replicate_attn:
            for name in _LTX2_ATTN_ATTRS:
                attn = getattr(block, name)
                if int(attn.heads) % int(tp_degree) != 0:
                    raise ValueError(
                        f"LTX-2 {name} heads {attn.heads} must divide tp={tp_degree}."
                    )
                if getattr(attn, "to_gate_logits", None) is not None:
                    raise NotImplementedError(
                        "LTX-2 tensor-parallel sharding does not support gated attention."
                    )
                attn.heads = int(attn.heads) // int(tp_degree)
                attn.inner_dim = int(attn.inner_dim) // int(tp_degree)
                attn.inner_kv_dim = int(attn.inner_kv_dim) // int(tp_degree)
                attn.to_q = _column_parallel_like(attn.to_q, gather_output=False)
                attn.to_k = _column_parallel_like(attn.to_k, gather_output=False)
                attn.to_v = _column_parallel_like(attn.to_v, gather_output=False)
                attn.to_out[0] = _row_parallel_like(
                    attn.to_out[0], input_is_parallel=True, reduce_dtype=reduce_dtype
                )
                attn.processor = processor

        if not replicate_mlp:
            block.ff.net[0].proj = _column_parallel_like(block.ff.net[0].proj, gather_output=False)
            block.ff.net[2] = _row_parallel_like(
                block.ff.net[2], input_is_parallel=True, reduce_dtype=reduce_dtype
            )
            block.audio_ff.net[0].proj = _column_parallel_like(
                block.audio_ff.net[0].proj, gather_output=False
            )
            block.audio_ff.net[2] = _row_parallel_like(
                block.audio_ff.net[2], input_is_parallel=True, reduce_dtype=reduce_dtype
            )

    # Slice RoPE once at the rope modules (heads were sharded above), so the
    # per-rank rope fans out to all blocks without re-scattering per attention.
    if not replicate_attn:
        _patch_ltx2_rope_for_tp(transformer, rank_util)


class _LTX2TransformerTraceModule(nn.Module):
    """Fix non-tensor LTX-2 transformer args at trace time."""

    def __init__(self, config):
        super().__init__()
        import diffusers.models.transformers.transformer_ltx2 as ltx2_transformer

        ltx2_transformer.apply_split_rotary_emb = _difflet_apply_split_rotary_emb
        LTX2VideoTransformer3DModel = ltx2_transformer.LTX2VideoTransformer3DModel

        self.config = config
        self.transformer = LTX2VideoTransformer3DModel(
            in_channels=int(config.in_channels),
            out_channels=int(config.out_channels),
            patch_size=int(config.patch_size),
            patch_size_t=int(config.patch_size_t),
            num_attention_heads=int(config.num_attention_heads),
            attention_head_dim=int(config.attention_head_dim),
            cross_attention_dim=int(config.cross_attention_dim),
            vae_scale_factors=tuple(config.vae_scale_factors),
            pos_embed_max_pos=int(getattr(config, "pos_embed_max_pos", 20)),
            base_height=int(getattr(config, "base_height", 2048)),
            base_width=int(getattr(config, "base_width", 2048)),
            gated_attn=bool(getattr(config, "gated_attn", False)),
            cross_attn_mod=bool(getattr(config, "cross_attn_mod", False)),
            audio_in_channels=int(config.audio_in_channels),
            audio_out_channels=int(config.audio_out_channels),
            audio_patch_size=int(config.audio_patch_size),
            audio_patch_size_t=int(config.audio_patch_size_t),
            audio_num_attention_heads=int(config.audio_num_attention_heads),
            audio_attention_head_dim=int(config.audio_attention_head_dim),
            audio_cross_attention_dim=int(config.audio_cross_attention_dim),
            audio_scale_factor=int(config.audio_scale_factor),
            audio_pos_embed_max_pos=int(getattr(config, "audio_pos_embed_max_pos", 20)),
            audio_sampling_rate=int(config.audio_sampling_rate),
            audio_hop_length=int(config.audio_hop_length),
            audio_gated_attn=bool(getattr(config, "audio_gated_attn", False)),
            audio_cross_attn_mod=bool(getattr(config, "audio_cross_attn_mod", False)),
            num_layers=int(config.num_layers),
            activation_fn=str(getattr(config, "activation_fn", "gelu-approximate")),
            qk_norm=str(getattr(config, "qk_norm", "rms_norm_across_heads")),
            norm_elementwise_affine=bool(getattr(config, "norm_elementwise_affine", False)),
            norm_eps=float(getattr(config, "norm_eps", 1e-6)),
            caption_channels=int(config.caption_channels),
            attention_bias=bool(getattr(config, "attention_bias", True)),
            attention_out_bias=bool(getattr(config, "attention_out_bias", True)),
            rope_theta=float(getattr(config, "rope_theta", 10000.0)),
            rope_double_precision=bool(getattr(config, "rope_double_precision", True)),
            causal_offset=int(getattr(config, "causal_offset", 1)),
            timestep_scale_multiplier=int(getattr(config, "timestep_scale_multiplier", 1000)),
            cross_attn_timestep_scale_multiplier=int(
                getattr(config, "cross_attn_timestep_scale_multiplier", 1000)
            ),
            rope_type=str(getattr(config, "rope_type", "interleaved")),
            use_prompt_embeddings=bool(getattr(config, "use_prompt_embeddings", True)),
            perturbed_attn=bool(getattr(config, "perturbed_attn", False)),
        )

        # CFG parallel: the caller stacks [uncond, cond] into batch=2 and one
        # branch is scattered to each data-parallel rank. STG/modality guidance
        # add extra batch!=2 transformer calls that can't run on this batch=2
        # graph, so they are rejected at the pipeline boundary; perturbed_attn
        # (STG) is rejected here for the same reason.
        self.cfg_parallel_enabled = bool(getattr(config, "cfg_parallel_enabled", False))
        if self.cfg_parallel_enabled:
            if bool(getattr(config, "perturbed_attn", False)):
                raise NotImplementedError(
                    "LTX-2 CFG-parallel does not support perturbed_attn (STG); "
                    "disable spatio-temporal guidance when cfg_parallel_enabled."
                )
            # The cfg axis is ONLY cond/uncond (size 2); STG was rejected above
            # so no extra guidance branch can leak onto it.
            init_parallel_mesh(config)
            self.cfg_group = get_cfg_group()
            self.global_rank = SPMDRank(world_size=get_world_group().size())

        # Tensor-parallel sharding: only when a TP group is live (device compile).
        # The runtime rank for RoPE/QK-norm head slicing comes from SPMDRank;
        # its buffer is populated via convert_hf_to_neuron_state_dict (arange).
        tp_degree = _safe_tensor_parallel_size()
        if tp_degree > 1:
            # The TP processor replaces the stock attention processor; it does not
            # implement perturbation (STG). Fail loudly rather than silently
            # dropping perturbation_mask/all_perturbed kwargs the block would pass.
            if bool(getattr(config, "perturbed_attn", False)):
                raise NotImplementedError(
                    "LTX-2 tensor-parallel sharding does not support perturbed_attn (STG)."
                )
            self.tp_rank_util = SPMDRank(tp_degree)
            _shard_ltx2_transformer(
                self.transformer,
                tp_degree,
                self.tp_rank_util,
                reduce_dtype=config_reduce_dtype(config),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        audio_encoder_attention_mask: torch.Tensor,
        video_coords: torch.Tensor,
        audio_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # CFG parallel: scatter the batch=2 [uncond, cond] stack so each
        # data-parallel rank denoises one branch at batch=1; the per-branch
        # outputs are gathered back into batch=2 below.
        if self.cfg_parallel_enabled:
            cfg_rank = get_cfg_rank_spmd(self.global_rank.get_rank())

            def _scatter(t: torch.Tensor) -> torch.Tensor:
                return scatter_to_process_group_spmd(
                    t, partition_dim=0, rank=cfg_rank,
                    process_group=self.cfg_group,
                )

            hidden_states = _scatter(hidden_states)
            audio_hidden_states = _scatter(audio_hidden_states)
            encoder_hidden_states = _scatter(encoder_hidden_states)
            audio_encoder_hidden_states = _scatter(audio_encoder_hidden_states)
            timestep = _scatter(timestep)
            sigma = _scatter(sigma)
            encoder_attention_mask = _scatter(encoder_attention_mask)
            audio_encoder_attention_mask = _scatter(audio_encoder_attention_mask)
            video_coords = _scatter(video_coords)
            audio_coords = _scatter(audio_coords)

        video_out, audio_out = self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            audio_timestep=timestep,
            sigma=sigma,
            audio_sigma=sigma,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=int(self.config.latent_num_frames),
            height=int(self.config.latent_height),
            width=int(self.config.latent_width),
            fps=float(getattr(self.config, "frame_rate", 24.0)),
            audio_num_frames=int(self.config.audio_num_frames),
            video_coords=video_coords,
            audio_coords=audio_coords,
            isolate_modalities=False,
            spatio_temporal_guidance_blocks=None,
            perturbation_mask=None,
            use_cross_timestep=bool(getattr(self.config, "use_cross_timestep", False)),
            return_dict=False,
        )

        if self.cfg_parallel_enabled:
            video_out = gather_from_tensor_model_parallel_region_with_dim(
                video_out, gather_dim=0, process_group=self.cfg_group,
            )
            audio_out = gather_from_tensor_model_parallel_region_with_dim(
                audio_out, gather_dim=0, process_group=self.cfg_group,
            )
        return video_out, audio_out


def _make_video_coords(
    *,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    patch_size_t: int,
    scale_factors: tuple[int, int, int],
    causal_offset: int,
    fps: float,
) -> torch.Tensor:
    frames = torch.arange(0, num_frames, patch_size_t, dtype=torch.float32)
    rows = torch.arange(0, height, patch_size, dtype=torch.float32)
    cols = torch.arange(0, width, patch_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(frames, rows, cols, indexing="ij"), dim=0)
    patch_size_tensor = torch.tensor(
        (patch_size_t, patch_size, patch_size),
        dtype=grid.dtype,
    )
    latent_coords = torch.stack(
        [grid, grid + patch_size_tensor.view(3, 1, 1, 1)],
        dim=-1,
    )
    latent_coords = latent_coords.flatten(1, 3).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    scale_tensor = torch.tensor(scale_factors, dtype=latent_coords.dtype)
    pixel_coords = latent_coords * scale_tensor.view(1, 3, 1, 1)
    pixel_coords[:, 0, ...] = (
        pixel_coords[:, 0, ...] + int(causal_offset) - int(scale_factors[0])
    ).clamp(min=0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
    return pixel_coords


def _make_audio_coords(
    *,
    batch_size: int,
    audio_num_frames: int,
    patch_size_t: int,
    scale_factor: int,
    causal_offset: int,
    sampling_rate: int,
    hop_length: int,
) -> torch.Tensor:
    coords = torch.arange(0, audio_num_frames, patch_size_t, dtype=torch.float32)
    start_mel = (coords * scale_factor + int(causal_offset) - int(scale_factor)).clamp(min=0)
    end_mel = (
        (coords + int(patch_size_t)) * scale_factor
        + int(causal_offset)
        - int(scale_factor)
    ).clamp(min=0)
    seconds_per_mel = float(hop_length) / float(sampling_rate)
    coords = torch.stack([start_mel * seconds_per_mel, end_mel * seconds_per_mel], dim=-1)
    return coords.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)


#: The processor's name before the extraction (it was Trainium-only then); kept for importers.
_LTX2TrainiumTPAttnProcessor = _LTX2TPAttnProcessor


__all__ = [
    "_LTX2_ATTN_ATTRS",
    "_LTX2TPAttnProcessor",
    "_LTX2TrainiumTPAttnProcessor",
    "_LTX2TransformerTraceModule",
    "_column_parallel_like",
    "_difflet_apply_split_rotary_emb",
    "_env_flag",
    "_make_audio_coords",
    "_make_video_coords",
    "_patch_ltx2_rope_for_tp",
    "_row_parallel_like",
    "_safe_tensor_parallel_size",
    "_shard_ltx2_transformer",
    "_tp_head_scatter",
]
