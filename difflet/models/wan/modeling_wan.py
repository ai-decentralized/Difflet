"""Wan 2.2 transformer modeling — hardware-agnostic.

This module is intentionally backend-neutral: it imports only from torch,
diffusers (pure-torch utility modules), and ``difflet.ops``. It must NOT import
from ``neuronx_distributed``, ``nkilib``, or ``torch_neuronx``.

Mapped from the diffusers reference at
``diffusers.models.transformers.transformer_wan.WanTransformer3DModel`` so a
state dict from the upstream HF checkpoint loads cleanly (parameter names and
shapes preserved).

Key Trainium-aware decisions:

* Q/K/V column-parallel projections use ``gather_output=False`` so attention is
  **head-sharded** across TP ranks — each rank computes its own ``heads/tp`` heads,
  and the output projection is a ``RowParallelLinear`` (all-reduce). Because the
  upstream ``qk_norm="rms_norm_across_heads"`` normalizes over the full ``inner_dim``,
  the sharded q/k RMSNorm is done in ``WanAttention._global_rms_norm``: a cross-rank
  sum-of-squares (``reduce_from_tensor_model_parallel_region``) gives the full-dim RMS
  denominator, and the norm weight is sliced to this rank's heads. One model-level
  ``SPMDRank`` (buffer populated by ``convert_backbone_state_dict``) provides the rank.
  At tp=1 / no group this degenerates to a plain ``RMSNorm`` (CPU reference path).
  (Earlier the projections gathered Q/K/V and ran the attention *replicated* on every
  rank — correct but ~2x slower; head-sharding pulls trn2 level with H100, parity
  cosine 0.9998 vs the replicated baseline.)
* Attention runs through ``difflet.ops.attention`` → the NKI ``attention_cte`` flash
  kernel (``_attn_kernel``), unmasked, ~4x faster than the prior PyTorch-SDPA fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.normalization import FP32LayerNorm

from difflet.ops import (
    ColumnParallelLinear,
    RMSNorm,
    RowParallelLinear,
    SPMDRank,
    apply_rotary_emb,
    attention,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region_with_dim,
    get_cfg_group,
    get_cfg_rank_spmd,
    get_cp_group,
    get_cp_rank_spmd,
    get_tensor_model_parallel_size,
    get_world_group,
    init_parallel_mesh,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    ring_attention,
    scatter_to_process_group_spmd,
)


def _safe_tp_size() -> int:
    """TP degree, or 1 when the tensor-parallel group is not initialized.

    ``modeling_wan`` is constructed both on the Neuron device (group live) and on bare
    CPU for the reference path / unit tests (no group). The raw
    ``get_tensor_model_parallel_size()`` asserts a live group, so the attention sharding
    must query the size through this guard and fall back to the unsharded tp=1 path.
    """
    try:
        return int(get_tensor_model_parallel_size())
    except Exception:
        return 1


def _sp_unbias(x: torch.Tensor, row_linear: nn.Module) -> torch.Tensor:
    """Correct the bias double-count of a ``reduce_output=False`` row-parallel
    linear under Megatron-SP.

    nxd's ``RowParallelLinear`` adds the *full* bias to each rank's **un-reduced**
    partial (``output_ + self.bias`` after the skipped reduce). The Megatron-SP
    ``ḡ`` reduce-scatter then sums across the TP group, so the bias lands ``tp×``
    instead of once. Subtract the ``(tp-1)×`` overcount so the bias is applied
    exactly once. No-op at ``tp == 1`` (CPU reference / unit tests), so the
    SP-vs-dense equivalence on CPU is unchanged.
    """
    bias = getattr(row_linear, "bias", None)
    if bias is None:
        return x
    tp = _safe_tp_size()
    if tp <= 1:
        return x
    return x - (tp - 1) * bias.to(x.dtype)


@dataclass
class WanTransformerConfig:
    """Hyperparameters for ``WanTransformer3DModel``.

    Defaults match ``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` config.json (each of
    the two transformer stages — ``transformer`` and ``transformer_2`` —
    shares this config in the 14B model).
    """

    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: Optional[int] = None
    added_kv_proj_dim: Optional[int] = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: Optional[int] = None
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.qk_norm != "rms_norm_across_heads":
            raise NotImplementedError(
                "Wan T2V spike currently supports only "
                "qk_norm='rms_norm_across_heads'."
            )
        unsupported = {
            "image_dim": self.image_dim,
            "added_kv_proj_dim": self.added_kv_proj_dim,
            "pos_embed_seq_len": self.pos_embed_seq_len,
        }
        active = {name: value for name, value in unsupported.items() if value is not None}
        if active:
            raise NotImplementedError(
                "Wan T2V spike does not support I2V/added-kv config fields: "
                + ", ".join(f"{name}={value!r}" for name, value in active.items())
            )

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def from_diffusers_dict(cls, raw: dict) -> "WanTransformerConfig":
        """Build a config from a diffusers ``transformer/config.json`` dict.

        Unknown keys are ignored so future config additions don't break us.
        """
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kept = {k: v for k, v in raw.items() if k in fields}
        if "patch_size" in kept and isinstance(kept["patch_size"], list):
            kept["patch_size"] = tuple(kept["patch_size"])
        return cls(**kept)


# ---------------------------------------------------------------------------
# 3D rotary positional embedding


class WanRotaryPosEmbed(nn.Module):
    """3D RoPE: independent T/H/W frequency grids concatenated per token.

    Mirrors ``diffusers...transformer_wan.WanRotaryPosEmbed`` exactly. The
    forward returns ``(freqs_cos, freqs_sin)``, each shaped
    ``(1, S_tokens, 1, attention_head_dim)``, ready to feed
    ``difflet.ops.apply_rotary_emb``.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float64
        freqs_cos: list[torch.Tensor] = []
        freqs_sin: list[torch.Tensor] = []
        for dim in (t_dim, h_dim, w_dim):
            cos, sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(cos)
            freqs_sin.append(sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf = num_frames // p_t
        pph = height // p_h
        ppw = width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]
        cos_t, cos_h, cos_w = self.freqs_cos.split(split_sizes, dim=1)
        sin_t, sin_h, sin_w = self.freqs_sin.split(split_sizes, dim=1)

        cos_f = cos_t[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        cos_y = cos_h[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        cos_x = cos_w[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        sin_f = sin_t[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        sin_y = sin_h[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        sin_x = sin_w[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([cos_f, cos_y, cos_x], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([sin_f, sin_y, sin_x], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        return freqs_cos, freqs_sin


# ---------------------------------------------------------------------------
# Time / text condition embedding


class WanTimeTextEmbedding(nn.Module):
    """T2V condition embedding: timestep + text projection.

    The HF reference's ``WanTimeTextImageEmbedding`` also handles I2V image
    conditioning. T2V is the spike's only target (cclogs/09 §1) so we drop
    the image branch. If we resurrect I2V in M2.5 the easy fix is wiring
    a ``WanImageEmbedding`` here behind an ``image_dim`` arg.
    """

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        target_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != target_dtype and target_dtype != torch.int8:
            timestep = timestep.to(target_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


# ---------------------------------------------------------------------------
# Feed-forward (Megatron pattern)


class WanFeedForward(nn.Module):
    """GELU(tanh) MLP with column/row parallel split.

    Up-projection sharded across TP via ColumnParallel(gather_output=False);
    activation is element-wise; down-projection is RowParallel with
    ``input_is_parallel=True`` so the cross-rank reduce happens inside the
    op.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        dtype: Optional[torch.dtype] = None,
        sp_enabled: bool = False,
    ):
        super().__init__()
        self.sp_enabled = sp_enabled
        self.net_in = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=True,
            gather_output=False,
            dtype=dtype,
            reduce_dtype=dtype,
        )
        self.net_out = RowParallelLinear(
            inner_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            dtype=dtype,
            reduce_dtype=dtype,
            # Under Megatron-SP the cross-rank sum is folded into the ḡ
            # reduce-scatter below, so the row-parallel must not also all-reduce.
            reduce_output=not sp_enabled,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Megatron-SP: x arrives sequence-sharded [B, S/tp, H]; gather to the full
        # sequence (g) for the column-parallel up-projection, then reduce-scatter
        # the row-parallel partial back to a sequence shard (ḡ).
        if self.sp_enabled:
            x = gather_from_sequence_parallel_region(x, dim=1)
        x = self.net_in(x)
        x = F.gelu(x, approximate="tanh")
        x = self.net_out(x)
        if self.sp_enabled:
            x = reduce_scatter_to_sequence_parallel_region(x, dim=1)
            x = _sp_unbias(x, self.net_out)
        return x


# ---------------------------------------------------------------------------
# Attention (self / cross)


def _attn_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    head_dim: int,
) -> torch.Tensor:
    """Run attention on (B, heads, S, head_dim) tensors via ``difflet.ops.attention``.

    Routes to the nkilib flash kernel (``attention_cte``) on Trainium — measured
    ~4x faster than the prior PyTorch-SDPA fallback (cclog 90: SDPA 15.33 ms vs
    attention_cte 3.69 ms at N=4096). Unmasked full attention, replicated across
    TP ranks; the M2.5 SDPA shortcut is retired. Mirrors the verified HV layout:
    flatten heads into the batch axis → (B*heads, S, head_dim), tp_q/tp_k in the
    standard (S, D) layout, no mask. The CPU reference (``difflet.ops`` cpu impl)
    computes the same standard attention, so trajectory parity is preserved.
    """
    batch, heads, seq_q, dim = q.shape
    seq_k = k.shape[2]
    q_flat = q.reshape(batch * heads, seq_q, dim)
    k_flat = k.reshape(batch * heads, seq_k, dim)
    v_flat = v.reshape(batch * heads, seq_k, dim)
    out = attention(
        q_flat,
        k_flat,
        v_flat,
        scale=1.0 / math.sqrt(head_dim),
        causal=False,
        attention_mask=None,
        tp_q=True,
        tp_k=True,
        tp_out=False,
    )
    return out.reshape(batch, heads, seq_q, dim)


class WanAttention(nn.Module):
    """Self- or cross-attention for WanTransformerBlock.

    Shape contract:
      hidden_states: (B, S_q, dim)
      encoder_hidden_states: (B, S_k, dim) when cross-attn else None
      rotary_emb: (cos, sin) each (1, S_q, 1, head_dim) for self-attn else None

    Returns: (B, S_q, dim).
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        eps: float = 1e-6,
        is_cross_attention: bool = False,
        dtype: Optional[torch.dtype] = None,
        context_parallel_enabled: bool = False,
        cp_mode: str = "gather_kv",
        sp_enabled: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.is_cross_attention = is_cross_attention
        self.context_parallel_enabled = context_parallel_enabled
        self.cp_mode = cp_mode
        self.sp_enabled = sp_enabled
        if context_parallel_enabled:
            self.cp_group = get_cp_group()
        inner_dim = heads * head_dim
        self.inner_dim = inner_dim

        # TP-aware head sharding: keep Q/K/V sharded across ranks (gather_output=False)
        # so each rank computes attention for its own 1/tp of the heads — the FFN is
        # already sharded, so this stops the attention being the one replicated stage.
        # The across-heads RMSNorm is then done with a cross-rank reduction in
        # _global_rms_norm (mirrors LTX-2). ``_rank_util`` (an SPMDRank) is injected by
        # the parent transformer after construction so the norm weight can be sliced to
        # this rank's heads; it stays None for a standalone/tp=1 build (full weight).
        self.tp_degree = _safe_tp_size()
        if heads % self.tp_degree != 0:
            raise ValueError(
                f"WanAttention heads={heads} must be divisible by tp={self.tp_degree}"
            )
        self.local_heads = heads // self.tp_degree
        self._rank_util = None

        self.to_q = ColumnParallelLinear(
            dim, inner_dim, bias=True, gather_output=False, dtype=dtype, reduce_dtype=dtype
        )
        self.to_k = ColumnParallelLinear(
            dim, inner_dim, bias=True, gather_output=False, dtype=dtype, reduce_dtype=dtype
        )
        self.to_v = ColumnParallelLinear(
            dim, inner_dim, bias=True, gather_output=False, dtype=dtype, reduce_dtype=dtype
        )
        self.to_out = nn.ModuleList(
            [
                # RowParallelLinear: input is the head-sharded attention output, so the
                # output projection sums each rank's partial contribution (all-reduce).
                RowParallelLinear(
                    inner_dim,
                    dim,
                    bias=True,
                    input_is_parallel=True,
                    dtype=dtype,
                    reduce_dtype=dtype,
                    # Under Megatron-SP the cross-rank sum is folded into the ḡ
                    # reduce-scatter in forward(); don't also all-reduce here.
                    reduce_output=not sp_enabled,
                ),
                nn.Identity(),  # diffusers reference has a Dropout here; inference path is no-op
            ]
        )

        # qk_norm operates on the full inner_dim before the head split
        # (qk_norm="rms_norm_across_heads" in the upstream config). The weight is loaded
        # full (inner_dim) and sliced to this rank's heads at runtime; the RMS denominator
        # is the full inner_dim via a cross-rank sum (see _global_rms_norm).
        self.norm_q = RMSNorm(inner_dim, eps=eps)
        self.norm_k = RMSNorm(inner_dim, eps=eps)

    def _global_rms_norm(self, norm: "RMSNorm", x: torch.Tensor) -> torch.Tensor:
        """``rms_norm_across_heads`` over the FULL inner_dim while ``x`` holds only this
        rank's head shard ``(B, S, inner_dim/tp)``.

        The sum-of-squares is reduced across the TP group so the RMS denominator is the
        full inner_dim (matching HF exactly), and the affine weight is sliced to this
        rank's heads. With ``_rank_util=None`` / ``tp=1`` the reduce and scatter are
        no-ops, so this is bit-identical to a plain ``RMSNorm`` — the path the CPU
        reference and unit tests exercise.
        """
        local_dim = x.shape[-1]
        local_sq = x.float().pow(2).sum(dim=-1, keepdim=True)
        global_sq = reduce_from_tensor_model_parallel_region(local_sq)
        full_dim = local_dim * self.tp_degree
        # difflet's RMSNorm is a CustomRMSNorm (eps stored as variance_epsilon).
        eps = getattr(norm, "eps", None) or getattr(norm, "variance_epsilon", None) or 1e-6
        x_normed = x.float() * torch.rsqrt(global_sq / full_dim + eps)
        weight = getattr(norm, "weight", None)
        if weight is not None:
            if self._rank_util is not None:
                weight = scatter_to_process_group_spmd(weight, 0, self._rank_util.get_rank(), None)
            x_normed = x_normed * weight.float()
        return x_normed.to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Megatron-SP: the image/query stream arrives sequence-sharded
        # [B, S/tp, H]; gather it to the full sequence (g) before the
        # column-parallel Q/K/V. For cross-attention the K,V come from the
        # text encoder_hidden_states, which is replicated (never sharded), so it
        # is left untouched.
        if self.sp_enabled:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, dim=1)

        kv_source = (
            encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        )

        q = self.to_q(hidden_states)
        k = self.to_k(kv_source)
        v = self.to_v(kv_source)

        q = self._global_rms_norm(self.norm_q, q)
        k = self._global_rms_norm(self.norm_k, k)

        q = q.unflatten(-1, (self.local_heads, self.head_dim))
        k = k.unflatten(-1, (self.local_heads, self.head_dim))
        v = v.unflatten(-1, (self.local_heads, self.head_dim))

        if rotary_emb is not None and not self.is_cross_attention:
            cos, sin = rotary_emb
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # (B, S, heads, dim) → (B, heads, S, dim) for the kernel.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # CP self-attention: ring rotates sharded K,V; gather_kv all-gathers full K,V.
        # Cross-attention K,V come from encoder_hidden_states which is not scattered.
        if self.context_parallel_enabled and not self.is_cross_attention and self.cp_mode == "ring":
            out = ring_attention(q, k, v, scale=1.0 / math.sqrt(self.head_dim), causal=False)
        else:
            if self.context_parallel_enabled and not self.is_cross_attention:
                stacked_kv = torch.stack([k, v], dim=0)  # [2, B, heads, S/cp, head_dim]
                stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                    stacked_kv, gather_dim=3, process_group=self.cp_group
                )  # [2, B, heads, S, head_dim]
                k, v = torch.unbind(stacked_kv, dim=0)
            out = _attn_kernel(q, k, v, head_dim=self.head_dim)

        # (B, local_heads, S, dim) → (B, S, local_heads * dim); RowParallelLinear then
        # all-reduces each rank's partial output projection back to the full model dim.
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        out = self.to_out[0](out)
        # Megatron-SP ḡ: reduce the row-parallel partial across the TP group and
        # scatter back to this rank's sequence shard [B, S/tp, H].
        if self.sp_enabled:
            out = reduce_scatter_to_sequence_parallel_region(out, dim=1)
            out = _sp_unbias(out, self.to_out[0])
        out = self.to_out[1](out)
        return out


# ---------------------------------------------------------------------------
# Transformer block


class WanTransformerBlock(nn.Module):
    """Self-attn → cross-attn → FFN with adaLN modulation.

    The modulation parameters (shift/scale/gate, ×2 for MSA and FFN) are
    summed from a per-block ``scale_shift_table`` and a per-step
    ``timestep_proj`` tensor, exactly matching the diffusers reference.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
        context_parallel_enabled: bool = False,
        cp_mode: str = "gather_kv",
        sp_enabled: bool = False,
    ):
        super().__init__()
        head_dim = dim // num_heads

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim, num_heads, head_dim, eps=eps, is_cross_attention=False, dtype=dtype,
            context_parallel_enabled=context_parallel_enabled, cp_mode=cp_mode,
            sp_enabled=sp_enabled,
        )

        self.attn2 = WanAttention(
            dim, num_heads, head_dim, eps=eps, is_cross_attention=True, dtype=dtype,
            context_parallel_enabled=context_parallel_enabled, cp_mode=cp_mode,
            sp_enabled=sp_enabled,
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        self.ffn = WanFeedForward(dim, ffn_dim, dtype=dtype, sp_enabled=sp_enabled)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        norm_h = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
            hidden_states
        )
        attn_out = self.attn1(norm_h, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_out * gate_msa).type_as(hidden_states)

        norm_h = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_out = self.attn2(norm_h, encoder_hidden_states, None)
        hidden_states = hidden_states + attn_out

        norm_h = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_out = self.ffn(norm_h)
        hidden_states = (hidden_states.float() + ff_out.float() * c_gate_msa).type_as(
            hidden_states
        )
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level transformer


class WanTransformer3DModel(nn.Module):
    """Wan 2.2 video DiT.

    forward args:
      hidden_states: (B, in_channels, T, H, W) noise latents.
      timestep: (B,) or (B, S_t) (Wan2.2 Ti2V variant uses the latter; T2V
        uses the former).
      encoder_hidden_states: (B, S_text, text_dim) text embeddings from
        umT5.
      timestep_seq_len: optional override when ``timestep.ndim==2``.

    returns:
      (B, out_channels, T, H, W) predicted noise.
    """

    def __init__(self, config: WanTransformerConfig, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.config = config
        self.dtype = dtype

        self.context_parallel_enabled = getattr(config, 'context_parallel_enabled', False)
        self.cp_mode = getattr(config, 'cp_mode', 'gather_kv')
        self.cfg_parallel_enabled = getattr(config, 'cfg_parallel_enabled', False)
        # Megatron-style sequence parallelism reuses the TP group; it is mutually
        # exclusive with CP (both shard the sequence dimension).
        self.sp_enabled = getattr(config, 'sp_enabled', False)
        if self.sp_enabled and self.context_parallel_enabled:
            raise ValueError(
                "sp_enabled and context_parallel_enabled are mutually exclusive "
                "(both shard the sequence dimension)."
            )
        if self.cfg_parallel_enabled and self.context_parallel_enabled:
            raise ValueError(
                "cfg_parallel_enabled and context_parallel_enabled are mutually "
                "exclusive (both consume the data-parallel lanes)."
            )
        # CFG parallel scatters the batch dim (uncond/cond) over the cfg axis;
        # CP scatters the sequence dim over the cp axis. Each collective fires
        # only in its own axis subgroup — the dp axis carries nothing here.
        # init_parallel_mesh must run before self.blocks is built: WanAttention
        # grabs get_cp_group() in its own __init__.
        if self.context_parallel_enabled or self.cfg_parallel_enabled:
            init_parallel_mesh(config)
            self.global_rank = SPMDRank(world_size=get_world_group().size())
        if self.cfg_parallel_enabled:
            self.cfg_group = get_cfg_group()
        if self.context_parallel_enabled:
            self.cp_group = get_cp_group()

        inner_dim = config.inner_dim

        self.rope = WanRotaryPosEmbed(
            attention_head_dim=config.attention_head_dim,
            patch_size=config.patch_size,
            max_seq_len=config.rope_max_seq_len,
            theta=config.rope_theta,
        )
        self.patch_embedding = nn.Conv3d(
            config.in_channels,
            inner_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        self.condition_embedder = WanTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=config.text_dim,
        )

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    dim=inner_dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_attention_heads,
                    cross_attn_norm=config.cross_attn_norm,
                    eps=config.eps,
                    dtype=dtype,
                    context_parallel_enabled=self.context_parallel_enabled,
                    cp_mode=self.cp_mode,
                    sp_enabled=self.sp_enabled,
                )
                for _ in range(config.num_layers)
            ]
        )

        # TP-aware attention sharding: one SPMDRank slices each block's across-heads
        # qk-norm weight to the local heads. Build it once (its ``.rank`` buffer is
        # populated by convert_backbone_state_dict → arange) and inject the reference
        # into every attention WITHOUT registering it as a submodule there (object
        # __setattr__), so the only buffer lives at the model root. Skipped at tp=1
        # (CPU / unit tests) — the attentions then use the plain-RMSNorm fallback.
        tp_degree = _safe_tp_size()
        self.tp_rank_util = None
        if tp_degree > 1:
            self.tp_rank_util = SPMDRank(tp_degree)
            for block in self.blocks:
                object.__setattr__(block.attn1, "_rank_util", self.tp_rank_util)
                object.__setattr__(block.attn2, "_rank_util", self.tp_rank_util)

        self.norm_out = FP32LayerNorm(inner_dim, config.eps, elementwise_affine=False)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size)
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    def _sp_seq_scatter(self, tensor: torch.Tensor, *, dim: int) -> torch.Tensor:
        """Megatron-SP forward-entry sequence scatter across the TP group.

        Uses the materialized SPMD rank buffer (``tp_rank_util``) rather than
        nxd's ``scatter_to_sequence_parallel_region``: under SPMD tracing the
        latter resolves ``group.rank()`` to a single constant, so every rank
        would keep the *same* chunk (verified on device: scatter→gather did not
        round-trip). ``scatter_to_process_group_spmd`` with the per-rank buffer is
        the same primitive the validated CP path and the qk-norm weight scatter
        use. Identity on CPU (``tp_rank_util is None`` / tp==1).
        """
        rank = self.tp_rank_util.get_rank() if self.tp_rank_util is not None else 0
        return scatter_to_process_group_spmd(
            tensor, partition_dim=dim, rank=rank, process_group=None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        # CFG parallel: the caller stacks [uncond, cond] into batch=2; scatter the
        # batch dim so each data-parallel rank denoises one branch at batch=1.
        if self.cfg_parallel_enabled:
            assert hidden_states.shape[0] == 2, (
                f"CFG parallel expects batch_size=2, got {hidden_states.shape[0]}"
            )
            assert timestep.shape[0] == 2, (
                f"CFG parallel expects batch_size=2, got {timestep.shape[0]}"
            )
            assert encoder_hidden_states.shape[0] == 2, (
                f"CFG parallel expects batch_size=2, got {encoder_hidden_states.shape[0]}"
            )
            cfg_rank = get_cfg_rank_spmd(self.global_rank.get_rank())
            hidden_states = scatter_to_process_group_spmd(
                hidden_states, partition_dim=0, rank=cfg_rank,
                process_group=self.cfg_group,
            )
            timestep = scatter_to_process_group_spmd(
                timestep, partition_dim=0, rank=cfg_rank,
                process_group=self.cfg_group,
            )
            encoder_hidden_states = scatter_to_process_group_spmd(
                encoder_hidden_states, partition_dim=0, rank=cfg_rank,
                process_group=self.cfg_group,
            )

        bs, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        ppf = num_frames // p_t
        pph = height // p_h
        ppw = width // p_w

        rotary_emb = self.rope(hidden_states)  # (cos, sin) each (1, S, 1, head_dim)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, S, inner_dim)

        if self.sp_enabled:
            # Megatron-SP: shard the patch sequence across the TP group so each
            # rank carries [B, S/tp, H] through the blocks. Rotary stays
            # full-sequence — attention gathers back internally before applying it.
            # NOTE: scatter MUST use the materialized SPMD rank buffer; nxd's
            # group.rank()-based sequence scatter is not per-rank under SPMD
            # tracing (every rank would keep the same chunk).
            hidden_states = self._sp_seq_scatter(hidden_states, dim=1)

        if self.context_parallel_enabled:
            cp_rank = get_cp_rank_spmd(self.global_rank.get_rank())
            hidden_states = scatter_to_process_group_spmd(
                hidden_states, partition_dim=1, rank=cp_rank,
                process_group=self.cp_group,
            )
            cos, sin = rotary_emb
            cos = scatter_to_process_group_spmd(
                cos, partition_dim=1, rank=cp_rank, process_group=self.cp_group,
            )
            sin = scatter_to_process_group_spmd(
                sin, partition_dim=1, rank=cp_rank, process_group=self.cp_group,
            )
            rotary_emb = (cos, sin)

        # timestep can be (B,) or (B, S_t) in Ti2V mode.
        if timestep.ndim == 2:
            ts_seq_len = timestep_seq_len if timestep_seq_len is not None else timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if self.sp_enabled and ts_seq_len is not None:
            # Per-token (Ti2V) modulation carries a sequence axis that must be
            # sharded to match the sequence-sharded hidden states. Broadcast
            # (non-per-token) modulation needs no scatter.
            timestep_proj = self._sp_seq_scatter(timestep_proj, dim=1)
            temb = self._sp_seq_scatter(temb, dim=1)

        for block in self.blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )

        if temb.ndim == 3:
            shift, scale = (
                self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (
                self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
            ).chunk(2, dim=1)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if self.sp_enabled:
            # Megatron-SP: re-assemble the full sequence from the per-rank shards
            # before un-patchifying.
            hidden_states = gather_from_sequence_parallel_region(hidden_states, dim=1)

        if self.context_parallel_enabled:
            hidden_states = gather_from_tensor_model_parallel_region_with_dim(
                hidden_states, gather_dim=1, process_group=self.cp_group,
            )

        hidden_states = hidden_states.reshape(bs, ppf, pph, ppw, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        # CFG parallel: gather the per-rank branches back into batch=2 so the
        # pipeline can apply the guidance formula on [uncond, cond].
        if self.cfg_parallel_enabled:
            output = gather_from_tensor_model_parallel_region_with_dim(
                output, gather_dim=0, process_group=self.cfg_group,
            )
        return output

    @torch.no_grad()
    def teacache_mod_input(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """TeaCache signal: block-0's modulated self-attention input (cclog 87).

        Replicates the prefix of ``forward`` up to ``blocks[0]``'s modulated norm
        input ``norm1(x)*(1+scale_msa)+shift_msa``. The modulation is timestep-only
        (``scale_shift_table + timestep_proj``); ``encoder_hidden_states`` is only
        needed because ``condition_embedder`` projects it (its result is unused
        here). The host TeaCache controller takes the relative-L1 of this tensor's
        step-to-step change as the skip signal (gate Pearson 0.99).
        """
        hs = self.patch_embedding(hidden_states)
        hs = hs.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep_seq_len if timestep_seq_len is not None else timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        _temb, timestep_proj, _ = self.condition_embedder(
            timestep, encoder_hidden_states, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        block0 = self.blocks[0]
        if timestep_proj.ndim == 4:
            shift_msa, scale_msa = (
                block0.scale_shift_table.unsqueeze(0) + timestep_proj.float()
            ).chunk(6, dim=2)[:2]
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
        else:
            shift_msa, scale_msa = (
                block0.scale_shift_table + timestep_proj.float()
            ).chunk(6, dim=1)[:2]

        return (block0.norm1(hs.float()) * (1 + scale_msa) + shift_msa).type_as(hs)


__all__ = [
    "WanAttention",
    "WanFeedForward",
    "WanRotaryPosEmbed",
    "WanTimeTextEmbedding",
    "WanTransformer3DModel",
    "WanTransformerBlock",
    "WanTransformerConfig",
]
