"""HunyuanVideo modeling helpers.

This module is backend-neutral. It may import torch, pure diffusers utility
layers, and ``difflet.ops`` only; Trainium-specific compile/load wrappers live
under ``difflet.backends.trainium``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings,
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
    RMSNorm,
)

from difflet.ops import (
    ColumnParallelLinear,
    RowParallelLinear,
    SPMDRank,
    apply_rotary_emb,
    attention,
    gather_from_tensor_model_parallel_region_with_dim,
    get_data_parallel_group,
    get_dp_rank_spmd,
    get_tensor_model_parallel_size,
    get_world_group,
    scatter_to_process_group_spmd,
)


@dataclass
class HunyuanVideoTransformerConfig:
    """Config for ``HunyuanVideoTransformer3DModel``.

    Defaults match the upstream diffusers HunyuanVideo transformer config.
    """

    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    num_layers: int = 20
    num_single_layers: int = 40
    num_refiner_layers: int = 2
    mlp_ratio: float = 4.0
    patch_size: int = 2
    patch_size_t: int = 1
    qk_norm: str = "rms_norm"
    guidance_embeds: bool = True
    text_embed_dim: int = 4096
    pooled_projection_dim: int = 768
    rope_theta: float = 256.0
    rope_axes_dim: tuple[int, ...] = (16, 56, 56)
    image_condition_type: str | None = None
    context_parallel_enabled: bool = False

    def __post_init__(self) -> None:
        if self.qk_norm != "rms_norm":
            raise NotImplementedError(
                "HunyuanVideo M3 currently supports only qk_norm='rms_norm'."
            )
        supported = {None, "latent_concat", "token_replace"}
        if self.image_condition_type not in supported:
            raise ValueError(
                "Invalid image_condition_type="
                f"{self.image_condition_type!r}; expected one of "
                f"{sorted(t for t in supported if t)}"
            )
        if self.image_condition_type == "token_replace":
            raise NotImplementedError(
                "HunyuanVideo M3 does not port token_replace conditioning yet."
            )

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def from_diffusers_dict(cls, raw: dict[str, Any]) -> "HunyuanVideoTransformerConfig":
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        kept = {key: value for key, value in raw.items() if key in fields}
        if "rope_axes_dim" in kept and isinstance(kept["rope_axes_dim"], list):
            kept["rope_axes_dim"] = tuple(kept["rope_axes_dim"])
        return cls(**kept)


class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return hidden_states.flatten(2).transpose(1, 2)


class HunyuanVideoAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int | None = None) -> None:
        super().__init__()
        out_features = out_features or 2 * in_features
        self.linear = ColumnParallelLinear(in_features, out_features, bias=True, gather_output=True)
        self.nonlinearity = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        return gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)


class HunyuanVideoGELU(nn.Module):
    """Diffusers-compatible GELU projection used by Hunyuan FeedForward."""

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=True, gather_output=False)
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate=self.approximate)


class HunyuanVideoLinearActivation(nn.Module):
    """Diffusers-compatible LinearActivation(..., activation='silu')."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=bias, gather_output=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.silu(hidden_states.float(), inplace=False).to(hidden_states.dtype)


class HunyuanVideoFeedForward(nn.Module):
    """Feed-forward layer with diffusers-compatible parameter names."""

    def __init__(self, dim: int, mult: float = 4.0, activation_fn: str = "gelu-approximate"):
        super().__init__()
        inner_dim = int(dim * mult)
        if activation_fn == "gelu-approximate":
            act_fn = HunyuanVideoGELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "linear-silu":
            act_fn = HunyuanVideoLinearActivation(dim, inner_dim)
        else:
            raise NotImplementedError(
                f"Unsupported HunyuanVideo FeedForward activation {activation_fn!r}"
            )
        self.net = nn.ModuleList(
            [
                act_fn,
                nn.Dropout(0.0),
                RowParallelLinear(inner_dim, dim, bias=True, input_is_parallel=True),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class HunyuanVideoSelfAttention(nn.Module):
    """Diffusers Attention-compatible self-attention for token refiner blocks."""

    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        dim_head: int,
        bias: bool = True,
    ):
        super().__init__()
        tp_degree = get_tensor_model_parallel_size()
        if heads % tp_degree != 0:
            raise ValueError(
                f"HunyuanVideoSelfAttention heads={heads} must be divisible by tp={tp_degree}"
            )
        self.heads = heads // tp_degree
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head
        self.to_q = ColumnParallelLinear(query_dim, self.inner_dim, bias=bias, gather_output=False)
        self.to_k = ColumnParallelLinear(query_dim, self.inner_dim, bias=bias, gather_output=False)
        self.to_v = ColumnParallelLinear(query_dim, self.inner_dim, bias=bias, gather_output=False)
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    self.inner_dim, query_dim, bias=True, input_is_parallel=True
                ),
                nn.Dropout(0.0),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("HunyuanVideoSelfAttention does not accept encoder_hidden_states")

        batch, sequence_length, _ = hidden_states.shape
        query = self.to_q(hidden_states).unflatten(2, (self.heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(2, (self.heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(2, (self.heads, self.head_dim))

        query = query.permute(0, 2, 1, 3).reshape(
            batch * self.heads, sequence_length, self.head_dim
        )
        key = key.permute(0, 2, 1, 3).reshape(batch * self.heads, sequence_length, self.head_dim)
        value = value.permute(0, 2, 1, 3).reshape(
            batch * self.heads, sequence_length, self.head_dim
        )
        mask = _flatten_attention_mask(attention_mask, batch=batch, heads=self.heads)

        hidden_states = attention(
            query,
            key,
            value,
            scale=1.0 / math.sqrt(self.head_dim),
            causal=False,
            attention_mask=mask,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )
        hidden_states = hidden_states.reshape(batch, self.heads, sequence_length, self.head_dim)
        hidden_states = hidden_states.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class HunyuanVideoAttention(nn.Module):
    """HunyuanVideo joint attention using the Difflet backend attention op."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        added_kv_proj_dim: int | None = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        context_parallel_enabled: bool = False,
    ):
        super().__init__()
        if qk_norm != "rms_norm":
            raise NotImplementedError("HunyuanVideo M3 currently supports qk_norm='rms_norm' only")

        self.context_parallel_enabled = context_parallel_enabled
        self.data_parallel_group = (
            get_data_parallel_group() if context_parallel_enabled else None
        )

        tp_degree = get_tensor_model_parallel_size()
        if num_attention_heads % tp_degree != 0:
            raise ValueError(
                "HunyuanVideoAttention num_attention_heads="
                f"{num_attention_heads} must be divisible by tp={tp_degree}"
            )
        self.heads = num_attention_heads // tp_degree
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.added_kv_proj_dim = added_kv_proj_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only

        self.to_q = ColumnParallelLinear(hidden_size, self.inner_dim, bias=True, gather_output=False)
        self.to_k = ColumnParallelLinear(hidden_size, self.inner_dim, bias=True, gather_output=False)
        self.to_v = ColumnParallelLinear(hidden_size, self.inner_dim, bias=True, gather_output=False)
        self.norm_q = RMSNorm(attention_head_dim, eps=eps)
        self.norm_k = RMSNorm(attention_head_dim, eps=eps)

        if added_kv_proj_dim is not None:
            self.add_k_proj = ColumnParallelLinear(
                added_kv_proj_dim, self.inner_dim, bias=True, gather_output=False
            )
            self.add_v_proj = ColumnParallelLinear(
                added_kv_proj_dim, self.inner_dim, bias=True, gather_output=False
            )
            if context_pre_only is not None:
                self.add_q_proj = ColumnParallelLinear(
                    added_kv_proj_dim, self.inner_dim, bias=True, gather_output=False
                )
            else:
                self.add_q_proj = None
            self.norm_added_q = RMSNorm(attention_head_dim, eps=eps)
            self.norm_added_k = RMSNorm(attention_head_dim, eps=eps)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None
            self.norm_added_q = None
            self.norm_added_k = None

        if not pre_only:
            self.to_out = nn.ModuleList(
                [
                    RowParallelLinear(
                        self.inner_dim,
                        hidden_size,
                        bias=True,
                        input_is_parallel=True,
                    ),
                    nn.Dropout(0.0),
                ]
            )
        else:
            self.to_out = None

        if context_pre_only is not None and not context_pre_only:
            self.to_add_out = RowParallelLinear(
                self.inner_dim,
                hidden_size,
                bias=True,
                input_is_parallel=True,
            )
        else:
            self.to_add_out = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.add_q_proj is None and encoder_hidden_states is not None:
            latent_seq = hidden_states.shape[1]
            joined = torch.cat([hidden_states, encoder_hidden_states], dim=1)
            query = self._shape(self.to_q(joined))
            key = self._shape(self.to_k(joined))
            value = self._shape(self.to_v(joined))
            query = self.norm_q(query)
            key = self.norm_k(key)

            latent_q, context_q = query[:, :latent_seq], query[:, latent_seq:]
            latent_k, context_k = key[:, :latent_seq], key[:, latent_seq:]
            latent_v, context_v = value[:, :latent_seq], value[:, latent_seq:]
            if image_rotary_emb is not None:
                latent_q = _apply_hunyuan_rotary(latent_q, image_rotary_emb)
                latent_k = _apply_hunyuan_rotary(latent_k, image_rotary_emb)
        else:
            if encoder_hidden_states is None:
                raise ValueError(
                    "HunyuanVideoAttention requires encoder_hidden_states for dual-stream attention"
                )
            latent_q = self.norm_q(self._shape(self.to_q(hidden_states)))
            latent_k = self.norm_k(self._shape(self.to_k(hidden_states)))
            latent_v = self._shape(self.to_v(hidden_states))
            if image_rotary_emb is not None:
                latent_q = _apply_hunyuan_rotary(latent_q, image_rotary_emb)
                latent_k = _apply_hunyuan_rotary(latent_k, image_rotary_emb)

            context_q = self._shape(self.add_q_proj(encoder_hidden_states))
            context_k = self._shape(self.add_k_proj(encoder_hidden_states))
            context_v = self._shape(self.add_v_proj(encoder_hidden_states))
            if self.norm_added_q is not None:
                context_q = self.norm_added_q(context_q)
            if self.norm_added_k is not None:
                context_k = self.norm_added_k(context_k)

        # Context parallel: each rank holds latent Q for its token shard. Gather
        # the latent K/V across the CP (data-parallel) group so every rank attends
        # over the full latent sequence. Latent Q stays sharded (q_len != kv_len).
        # Context (text) K/V are replicated, so they are left untouched.
        if self.context_parallel_enabled:
            stacked_kv = torch.stack([latent_k, latent_v], dim=0)  # [2, B, S/cp, H, d]
            stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                stacked_kv, gather_dim=2, process_group=self.data_parallel_group
            )  # [2, B, S, H, d]
            latent_k, latent_v = torch.unbind(stacked_kv, dim=0)

        hidden_states, encoder_hidden_states = dual_stream_attention(
            latent_q,
            latent_k,
            latent_v,
            context_q,
            context_k,
            context_v,
            attention_mask=attention_mask,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        hidden_states = hidden_states.flatten(2, 3).to(latent_q.dtype)
        encoder_hidden_states = encoder_hidden_states.flatten(2, 3).to(latent_q.dtype)

        if self.to_out is not None:
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
        if self.to_add_out is not None:
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states

    def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unflatten(2, (self.heads, self.head_dim))


class HunyuanVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
        context_parallel_enabled: bool = False,
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.attn = HunyuanVideoAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            added_kv_proj_dim=hidden_size,
            context_pre_only=False,
            pre_only=False,
            qk_norm=qk_norm,
            eps=1e-6,
            context_parallel_enabled=context_parallel_enabled,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = HunyuanVideoFeedForward(hidden_size, mult=mlp_ratio)
        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = HunyuanVideoFeedForward(hidden_size, mult=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        (
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1_context(
            encoder_hidden_states,
            emb=temb,
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = (
            encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)
        )

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(
            norm_encoder_hidden_states
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
        context_parallel_enabled: bool = False,
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)
        self.attn = HunyuanVideoAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            added_kv_proj_dim=None,
            context_pre_only=None,
            pre_only=True,
            qk_norm=qk_norm,
            eps=1e-6,
            context_parallel_enabled=context_parallel_enabled,
        )
        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = ColumnParallelLinear(hidden_size, mlp_dim, bias=True, gather_output=False)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = RowParallelLinear(
            hidden_size + mlp_dim,
            hidden_size,
            bias=True,
            input_is_parallel=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        residual = hidden_states

        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states[:, :-text_seq_length, :], hidden_states[:, -text_seq_length:, :]


class HunyuanVideoConditionEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        guidance_embeds: bool,
        image_condition_type: str | None = None,
    ):
        super().__init__()
        self.image_condition_type = image_condition_type
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu"
        )
        self.guidance_embedder = None
        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(
                in_channels=256, time_embed_dim=embedding_dim
            )

    def forward(
        self,
        timestep: torch.Tensor,
        pooled_projection: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)

        token_replace_emb = None
        if self.image_condition_type == "token_replace":
            token_replace_timestep = torch.zeros_like(timestep)
            token_replace_proj = self.time_proj(token_replace_timestep)
            token_replace_emb = self.timestep_embedder(
                token_replace_proj.to(dtype=pooled_projection.dtype)
            )
            token_replace_emb = token_replace_emb + pooled_projections

        if self.guidance_embedder is not None:
            if guidance is None:
                raise ValueError("guidance must be provided when guidance_embeds=True")
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
            conditioning = timesteps_emb + guidance_emb + pooled_projections
        else:
            conditioning = timesteps_emb + pooled_projections
        return conditioning, token_replace_emb


class HunyuanVideoIndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = HunyuanVideoSelfAttention(
            query_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = HunyuanVideoFeedForward(
            hidden_size,
            mult=mlp_width_ratio,
            activation_fn="linear-silu",
        )
        if mlp_drop_rate != 0.0:
            self.ff.net[1] = nn.Dropout(mlp_drop_rate)
        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp
        return hidden_states


class HunyuanVideoIndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideoIndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(
                1, 1, seq_len, 1
            )
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask[:, :, :, 0] = True

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)
        return hidden_states


class HunyuanVideoTokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size,
            pooled_projection_dim=in_channels,
        )
        self.proj_in = ColumnParallelLinear(in_channels, hidden_size, bias=True, gather_output=True)
        self.token_refiner = HunyuanVideoIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        return self.token_refiner(hidden_states, temb, attention_mask)


class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_size_t: int,
        rope_dim: tuple[int, ...],
        theta: float = 256.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, num_frames, height, width = hidden_states.shape
        rope_sizes = [
            num_frames // self.patch_size_t,
            height // self.patch_size,
            width // self.patch_size,
        ]

        axes_grids = []
        for size in rope_sizes:
            axes_grids.append(
                torch.arange(0, size, device=hidden_states.device, dtype=torch.float32)
            )
        grid = torch.meshgrid(*axes_grids, indexing="ij")
        grid = torch.stack(grid, dim=0)

        freqs = []
        for i in range(3):
            freqs.append(
                get_1d_rotary_pos_embed(
                    self.rope_dim[i],
                    grid[i].reshape(-1),
                    self.theta,
                    use_real=True,
                )
            )

        freqs_cos = torch.cat([freq[0] for freq in freqs], dim=1)
        freqs_sin = torch.cat([freq[1] for freq in freqs], dim=1)
        return freqs_cos, freqs_sin


class HunyuanVideoTransformer3DModel(nn.Module):
    _supports_gradient_checkpointing = False
    _no_split_modules = [
        "HunyuanVideoTransformerBlock",
        "HunyuanVideoSingleTransformerBlock",
        "HunyuanVideoPatchEmbed",
        "HunyuanVideoTokenRefiner",
    ]

    def __init__(
        self,
        in_channels: int | HunyuanVideoTransformerConfig = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: tuple[int, ...] = (16, 56, 56),
        image_condition_type: str | None = None,
        context_parallel_enabled: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(in_channels, int):
            config = in_channels
            context_parallel_enabled = getattr(config, "context_parallel_enabled", False)
            in_channels = config.in_channels
            out_channels = config.out_channels
            num_attention_heads = config.num_attention_heads
            attention_head_dim = config.attention_head_dim
            num_layers = config.num_layers
            num_single_layers = config.num_single_layers
            num_refiner_layers = config.num_refiner_layers
            mlp_ratio = config.mlp_ratio
            patch_size = config.patch_size
            patch_size_t = config.patch_size_t
            qk_norm = config.qk_norm
            guidance_embeds = config.guidance_embeds
            text_embed_dim = config.text_embed_dim
            pooled_projection_dim = config.pooled_projection_dim
            rope_theta = config.rope_theta
            rope_axes_dim = tuple(config.rope_axes_dim)
            image_condition_type = getattr(config, "image_condition_type", None)

        self.config = HunyuanVideoTransformerConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            num_refiner_layers=num_refiner_layers,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            qk_norm=qk_norm,
            guidance_embeds=guidance_embeds,
            text_embed_dim=text_embed_dim,
            pooled_projection_dim=pooled_projection_dim,
            rope_theta=rope_theta,
            rope_axes_dim=rope_axes_dim,
            image_condition_type=image_condition_type,
            context_parallel_enabled=context_parallel_enabled,
        )

        inner_dim = self.config.inner_dim
        out_channels = out_channels or in_channels

        self.context_parallel_enabled = context_parallel_enabled
        if context_parallel_enabled:
            self.data_parallel_group = get_data_parallel_group()
            self.global_rank = SPMDRank(world_size=get_world_group().size())

        self.x_embedder = HunyuanVideoPatchEmbed(
            (patch_size_t, patch_size, patch_size), in_channels, inner_dim
        )
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim,
            num_attention_heads,
            attention_head_dim,
            num_layers=num_refiner_layers,
        )
        self.time_text_embed = HunyuanVideoConditionEmbedding(
            inner_dim,
            pooled_projection_dim,
            guidance_embeds,
            image_condition_type,
        )
        self.rope = HunyuanVideoRotaryPosEmbed(patch_size, patch_size_t, rope_axes_dim, rope_theta)
        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    context_parallel_enabled=context_parallel_enabled,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoSingleTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    context_parallel_enabled=context_parallel_enabled,
                )
                for _ in range(num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = ColumnParallelLinear(
            inner_dim,
            patch_size_t * patch_size * patch_size * out_channels,
            bias=True,
            gather_output=True,
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        del attention_kwargs
        batch_size, _, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        image_rotary_emb = self.rope(hidden_states)
        temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)
        if token_replace_emb is not None:
            raise NotImplementedError(
                "token_replace conditioning is not enabled in HunyuanVideo M3"
            )

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
        )

        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.ones(
            batch_size,
            sequence_length,
            device=hidden_states.device,
            dtype=torch.bool,
        )
        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
        indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
        mask_indices = indices >= effective_sequence_length.unsqueeze(1)
        attention_mask = attention_mask.masked_fill(mask_indices, False)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        # Context parallel: scatter the latent tokens (and their rotary embeddings)
        # along the sequence axis across the CP (data-parallel) group. The text
        # tokens and the key-axis attention mask stay full/replicated; the mask is
        # intentionally built above with the full latent length so it still lines up
        # with the gathered K/V inside attention.
        if self.context_parallel_enabled:
            dp_rank = get_dp_rank_spmd(
                global_rank=self.global_rank.get_rank(),
                tp_degree=get_tensor_model_parallel_size(),
            )
            hidden_states = scatter_to_process_group_spmd(
                hidden_states,
                partition_dim=1,
                rank=dp_rank,
                process_group=self.data_parallel_group,
            )
            cos, sin = image_rotary_emb
            cos = scatter_to_process_group_spmd(
                cos, partition_dim=0, rank=dp_rank, process_group=self.data_parallel_group
            )
            sin = scatter_to_process_group_spmd(
                sin, partition_dim=0, rank=dp_rank, process_group=self.data_parallel_group
            )
            image_rotary_emb = (cos, sin)

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Context parallel: reassemble the full latent sequence before unpatching.
        if self.context_parallel_enabled:
            hidden_states = gather_from_tensor_model_parallel_region_with_dim(
                hidden_states, gather_dim=1, process_group=self.data_parallel_group
            )

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p,
            p,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)

    def teacache_mod_input(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the timestep-modulated latent input to transformer block 0."""
        del encoder_hidden_states, encoder_attention_mask
        temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)
        if token_replace_emb is not None:
            raise NotImplementedError(
                "token_replace conditioning is not enabled in HunyuanVideo M3"
            )
        hidden_states = self.x_embedder(hidden_states)
        norm_hidden_states, *_ = self.transformer_blocks[0].norm1(hidden_states, emb=temb)
        return norm_hidden_states


def dual_stream_attention(
    latent_q: torch.Tensor,
    latent_k: torch.Tensor,
    latent_v: torch.Tensor,
    context_q: torch.Tensor,
    context_k: torch.Tensor,
    context_v: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HunyuanVideo joint attention over latent and context streams.

    Inputs follow diffusers' HunyuanVideo processor shape:
    ``(batch, sequence, heads, head_dim)``. The implementation concatenates
    latent and context Q/K/V along sequence, runs one non-causal attention via
    ``difflet.ops.attention``, and splits the result back into the two streams.

    Under context parallelism the latent query is sharded across CP ranks while
    the latent K/V have already been gathered to the full sequence, so the latent
    query length may be shorter than the latent K/V length (``q_len != kv_len``).
    """
    _check_stream_shapes(context_q, context_k, context_v, name="context")
    # latent K/V must match each other; latent Q may be a shorter shard (CP), so
    # only its batch/heads/head_dim must agree with the latent K/V and context.
    if latent_k.shape != latent_v.shape:
        raise ValueError(
            "latent K and V must share shape: "
            f"k={tuple(latent_k.shape)} v={tuple(latent_v.shape)}"
        )
    if latent_q.ndim != 4 or latent_k.ndim != 4:
        raise ValueError("latent Q/K/V must be 4D (batch, sequence, heads, head_dim)")
    if latent_q.shape[0] != latent_k.shape[0] or latent_q.shape[2:] != latent_k.shape[2:]:
        raise ValueError(
            "latent Q must share batch, heads, and head_dim with latent K/V: "
            f"q={tuple(latent_q.shape)} k={tuple(latent_k.shape)}"
        )
    if latent_q.shape[0] != context_q.shape[0] or latent_q.shape[2:] != context_q.shape[2:]:
        raise ValueError(
            "latent and context streams must share batch, heads, and head_dim: "
            f"latent={tuple(latent_q.shape)} context={tuple(context_q.shape)}"
        )

    batch, q_latent_seq, heads, head_dim = latent_q.shape
    kv_latent_seq = latent_k.shape[1]
    context_seq = context_q.shape[1]
    q_len = q_latent_seq + context_seq
    kv_len = kv_latent_seq + context_seq
    scale = (1.0 / math.sqrt(head_dim)) if scale is None else scale

    q = torch.cat([latent_q, context_q], dim=1)
    k = torch.cat([latent_k, context_k], dim=1)
    v = torch.cat([latent_v, context_v], dim=1)

    q_flat = q.permute(0, 2, 1, 3).reshape(batch * heads, q_len, head_dim)
    k_flat = k.permute(0, 2, 1, 3).reshape(batch * heads, kv_len, head_dim)
    v_flat = v.permute(0, 2, 1, 3).reshape(batch * heads, kv_len, head_dim)
    mask_flat = _flatten_attention_mask(attention_mask, batch=batch, heads=heads)

    if mask_flat is not None:
        # Route the contiguous (right-padded) key-padding mask to attention_cte's
        # lossless bound_min/bound_max range instead of the SDPA fallback. SDPA
        # materializes the full q_len x kv_len scores+mask, which is catastrophic for
        # this ~40k-token joint video sequence (the measured ~4-5x slowdown, i.e. the
        # model looked like it had no attention_cte); the bound path is flash-style.
        # Bounds are derived with a plain sum (trace-safe — mirrors the model
        # forward's encoder_attention_mask.sum), NOT the general in-graph
        # mask_to_contiguous_bounds, which is unsafe to trace (see commit cd54d0f /
        # ops_impl/attention.py).
        bound_min, bound_max = _keypad_bounds_from_mask(mask_flat, q_len)
        out = attention(
            q_flat,
            k_flat,
            v_flat,
            scale=scale,
            causal=False,
            bound_min=bound_min,
            bound_max=bound_max,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )
    else:
        out = attention(
            q_flat,
            k_flat,
            v_flat,
            scale=scale,
            causal=False,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )
    out = out.reshape(batch, heads, q_len, head_dim).permute(0, 2, 1, 3)
    return out[:, :q_latent_seq], out[:, q_latent_seq:]


def _keypad_bounds_from_mask(
    mask_flat: torch.Tensor, q_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded key-padding mask -> attention_cte contiguous ``[0, count)`` bounds.

    HunyuanVideo's joint DiT mask (built in ``HunyuanVideoTransformer3DModel.forward``)
    marks the valid keys as the contiguous prefix ``[0, n_latent + n_valid_text)`` and
    is uniform across queries, so the valid set is losslessly an attention_cte
    ``bound_min``/``bound_max`` range: ``lo=0``, ``hi=#valid keys``. Counting valid keys
    with a plain sum is trace-safe (it mirrors the forward's
    ``encoder_attention_mask.sum``); the general ``mask_to_contiguous_bounds`` must NOT
    run in-graph (XLA broadcast error — see ``ops_impl/attention.py`` and commit
    cd54d0f).

    ``mask_flat``: ``(B*heads, query_or_1, kv_len)``, bool (``True``=attend) or {0,1}.
    Returns ``(bound_min, bound_max)`` int32 of shape ``(B*heads, q_len, 1)``.
    """
    bh = mask_flat.shape[0]
    valid = mask_flat if mask_flat.dtype == torch.bool else (mask_flat != 0)
    # sum() promotes to int64; attention_cte wants int32 bounds (cf. mask_bounds.py).
    count = valid.sum(dim=-1, keepdim=True).to(torch.int32)   # (B*heads, q_or_1, 1)
    bound_max = count.expand(bh, q_len, 1).contiguous()       # (B*heads, q_len, 1)
    bound_min = torch.zeros_like(bound_max)
    return bound_min, bound_max


def _check_stream_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, name: str) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"{name} Q/K/V must be 4D (batch, sequence, heads, head_dim)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"{name} Q/K/V shapes must match: q={tuple(q.shape)} "
            f"k={tuple(k.shape)} v={tuple(v.shape)}"
        )


def _flatten_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch: int,
    heads: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 4 or attention_mask.shape[0] != batch:
        raise ValueError(
            "attention_mask must be shaped like diffusers HunyuanVideo mask "
            f"(batch, 1, query_or_1, key); got {tuple(attention_mask.shape)}"
        )
    expanded = attention_mask.expand(
        batch, heads, attention_mask.shape[-2], attention_mask.shape[-1]
    )
    return expanded.reshape(batch * heads, attention_mask.shape[-2], attention_mask.shape[-1])


def _apply_hunyuan_rotary(
    hidden_states: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos, sin = image_rotary_emb
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
    if sin.ndim == 2:
        sin = sin.unsqueeze(0).unsqueeze(2)
    return apply_rotary_emb(
        hidden_states,
        cos.to(hidden_states.device),
        sin.to(hidden_states.device),
    )
