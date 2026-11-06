"""Instance-local HunyuanVideo CUDA attention and compact bias preparation.

The shared model and its Trainium/Neuron attention methods remain unchanged.
CUDA supports one unsharded stream and prepares the additive bias before the
regionally compiled blocks, keeping mask storage linear in sequence length.
"""

from __future__ import annotations

import math
import types
from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from difflet.models.hunyuan_video.modeling_hunyuan_video import (
    HunyuanVideoAttention,
    HunyuanVideoTransformer3DModel,
    _apply_hunyuan_rotary,
    _check_stream_shapes,
    _flatten_attention_mask,
)
from difflet.backends.cuda.ops_impl.attention import attention


def _prepare_compact_joint_bias(keep_mask, dtype):
    return torch.zeros_like(keep_mask, dtype=dtype).masked_fill(~keep_mask, float("-inf"))


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
    if latent_k.shape != latent_v.shape:
        raise ValueError(
            f"latent K and V must share shape: k={tuple(latent_k.shape)} v={tuple(latent_v.shape)}"
        )
    if latent_q.ndim != 4 or latent_k.ndim != 4:
        raise ValueError("latent Q/K/V must be 4D (batch, sequence, heads, head_dim)")
    if latent_q.shape[0] != latent_k.shape[0] or latent_q.shape[2:] != latent_k.shape[2:]:
        raise ValueError(
            f"latent Q must share batch, heads, and head_dim with latent K/V: q={tuple(latent_q.shape)} k={tuple(latent_k.shape)}"
        )
    if latent_q.shape[0] != context_q.shape[0] or latent_q.shape[2:] != context_q.shape[2:]:
        raise ValueError(
            f"latent and context streams must share batch, heads, and head_dim: latent={tuple(latent_q.shape)} context={tuple(context_q.shape)}"
        )
    batch, q_latent_seq, heads, head_dim = latent_q.shape
    kv_latent_seq = latent_k.shape[1]
    context_seq = context_q.shape[1]
    q_len = q_latent_seq + context_seq
    kv_len = kv_latent_seq + context_seq
    scale = 1.0 / math.sqrt(head_dim) if scale is None else scale
    q = torch.cat([latent_q, context_q], dim=1)
    k = torch.cat([latent_k, context_k], dim=1)
    v = torch.cat([latent_v, context_v], dim=1)
    q_flat = q.permute(0, 2, 1, 3).reshape(batch * heads, q_len, head_dim)
    k_flat = k.permute(0, 2, 1, 3).reshape(batch * heads, kv_len, head_dim)
    v_flat = v.permute(0, 2, 1, 3).reshape(batch * heads, kv_len, head_dim)
    mask_flat = _flatten_attention_mask(attention_mask, batch=batch, heads=heads)
    out = attention(
        q_flat,
        k_flat,
        v_flat,
        scale=scale,
        causal=False,
        attention_mask=mask_flat,
        tp_q=True,
        tp_k=True,
        tp_out=False,
    )
    out = out.reshape(batch, heads, q_len, head_dim).permute(0, 2, 1, 3)
    return (out[:, :q_latent_seq], out[:, q_latent_seq:])


def _attention_forward(
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
        latent_q, context_q = (query[:, :latent_seq], query[:, latent_seq:])
        latent_k, context_k = (key[:, :latent_seq], key[:, latent_seq:])
        latent_v, context_v = (value[:, :latent_seq], value[:, latent_seq:])
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
    return (hidden_states, encoder_hidden_states)


class CudaHunyuanVideoTransformer3DModel(HunyuanVideoTransformer3DModel):
    def __init__(self, config):
        super().__init__(config)
        self.joint_attention_mask_mode = config.joint_attention_mask_mode
        if config.joint_attention_mask_mode == "compact":
            for layer in self.modules():
                if isinstance(layer, HunyuanVideoAttention):
                    layer.forward = types.MethodType(_attention_forward, layer)

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
        p, p_t = (self.config.patch_size, self.config.patch_size_t)
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
            encoder_hidden_states, timestep, encoder_attention_mask
        )
        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.ones(
            batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
        )
        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
        indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
        mask_indices = indices >= effective_sequence_length.unsqueeze(1)
        attention_mask = attention_mask.masked_fill(mask_indices, False)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        if self.joint_attention_mask_mode == "compact":
            attention_mask = _prepare_compact_joint_bias(attention_mask, hidden_states.dtype)
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
            )
        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
            )
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)
