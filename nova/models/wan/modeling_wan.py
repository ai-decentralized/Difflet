"""Wan 2.2 transformer modeling — hardware-agnostic.

This module is intentionally backend-neutral: it imports only from torch,
diffusers (pure-torch utility modules), and ``nova.ops``. It must NOT import
from ``neuronx_distributed``, ``nkilib``, or ``torch_neuronx``.

Mapped from the diffusers reference at
``diffusers.models.transformers.transformer_wan.WanTransformer3DModel`` so a
state dict from the upstream HF checkpoint loads cleanly (parameter names and
shapes preserved).

Key Trainium-aware decisions:

* Q/K/V column-parallel projections use ``gather_output=True`` so q/k RMSNorm
  runs on the full ``inner_dim`` (matches the reference's
  ``rms_norm_across_heads`` semantics). The spike keeps attention replicated
  across TP ranks and uses ``ColumnParallelLinear(gather_output=True)`` for the
  attention output projection, matching the UMT5 replicated-attention pattern
  that is already covered by NEFF-vs-CPU. This trades attention compute for a
  rank-safe numerical reference path. A future TP-aware RMSNorm can shard Q/K/V
  before attention for full TP efficiency.
* Attention uses PyTorch SDPA for the M2.5 numerical path. This is slower than
  the Flux-style NKI kernel path but gives a direct CPU/NEFF semantic match
  while Wan-specific TP sharding is still being validated.
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

from nova.ops import (
    ColumnParallelLinear,
    RMSNorm,
    RowParallelLinear,
    apply_rotary_emb,
)


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
    ``nova.ops.apply_rotary_emb``.
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

    def __init__(self, dim: int, inner_dim: int, dtype: Optional[torch.dtype] = None):
        super().__init__()
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net_in(x)
        x = F.gelu(x, approximate="tanh")
        x = self.net_out(x)
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
    """Run attention on (B, heads, S, head_dim) tensors.

    The M2.5 numerical path intentionally uses PyTorch SDPA instead of the NKI
    attention kernel. This keeps the traced NEFF graph on the same high-level
    semantics as the CPU reference while Wan TP sharding is still being
    stabilized.
    """
    scale = 1.0 / math.sqrt(head_dim)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )


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
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.is_cross_attention = is_cross_attention
        inner_dim = heads * head_dim

        # gather_output=True → full inner_dim on every rank so RMSNorm
        # over the full feature axis matches the HF reference exactly.
        # See module docstring for the trade-off.
        self.to_q = ColumnParallelLinear(
            dim, inner_dim, bias=True, gather_output=True, dtype=dtype, reduce_dtype=dtype
        )
        self.to_k = ColumnParallelLinear(
            dim, inner_dim, bias=True, gather_output=True, dtype=dtype, reduce_dtype=dtype
        )
        self.to_v = ColumnParallelLinear(
            dim, inner_dim, bias=True, gather_output=True, dtype=dtype, reduce_dtype=dtype
        )
        self.to_out = nn.ModuleList(
            [
                ColumnParallelLinear(
                    inner_dim,
                    dim,
                    bias=True,
                    gather_output=True,
                    dtype=dtype,
                    reduce_dtype=dtype,
                ),
                nn.Identity(),  # diffusers reference has a Dropout here; inference path is no-op
            ]
        )

        # qk_norm operates on the full inner_dim before head split
        # (qk_norm="rms_norm_across_heads" in the upstream config).
        self.norm_q = RMSNorm(inner_dim, eps=eps)
        self.norm_k = RMSNorm(inner_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        kv_source = (
            encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        )

        q = self.to_q(hidden_states)
        k = self.to_k(kv_source)
        v = self.to_v(kv_source)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.unflatten(-1, (self.heads, self.head_dim))
        k = k.unflatten(-1, (self.heads, self.head_dim))
        v = v.unflatten(-1, (self.heads, self.head_dim))

        if rotary_emb is not None and not self.is_cross_attention:
            cos, sin = rotary_emb
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # (B, S, heads, dim) → (B, heads, S, dim) for the kernel. Attention is
        # intentionally replicated across TP ranks for the M2.5 numerical path.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = _attn_kernel(q, k, v, head_dim=self.head_dim)

        # (B, heads, S, dim) → (B, S, heads * dim), ready for a replicated
        # ColumnParallelLinear(gather_output=True) output projection.
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        out = self.to_out[0](out)
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
    ):
        super().__init__()
        head_dim = dim // num_heads

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim, num_heads, head_dim, eps=eps, is_cross_attention=False, dtype=dtype
        )

        self.attn2 = WanAttention(
            dim, num_heads, head_dim, eps=eps, is_cross_attention=True, dtype=dtype
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        self.ffn = WanFeedForward(dim, ffn_dim, dtype=dtype)
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
                )
                for _ in range(config.num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(inner_dim, config.eps, elementwise_affine=False)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size)
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        bs, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        ppf = num_frames // p_t
        pph = height // p_h
        ppw = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, S, inner_dim)

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

        hidden_states = hidden_states.reshape(bs, ppf, pph, ppw, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

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
