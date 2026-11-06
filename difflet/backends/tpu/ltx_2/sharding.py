"""LTX-2 on the TPU backend: build diffusers' transformer and shard it for TP.

The sharding recipe is the backend-neutral one in ``difflet/models/ltx_2/modeling_ltx_2.py``
(head-sharded attention with a global qk RMS norm, per-rank RoPE, column/row-parallel FFN),
used as is. Two things are TPU-only and live here:

* ``build_ltx2_transformer``: the shared module builds the transformer inside the Trainium
  trace module; the TPU port wants the bare diffusers model from a difflet config.
* ``TpuLTX2AttnProcessor``: the TPU attention op takes ``bound_min`` / ``bound_max``
  key windows for cross-attention, so the text padding mask is honored here (Trainium's
  kernel cannot take bounds for q_len != kv_len and runs cross-attention unmasked).
  Parity against the masked-SDPA reference went 0.9932 -> 0.99983 with it (ed0d1987).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from difflet.models.ltx_2.modeling_ltx_2 import (
    _LTX2_ATTN_ATTRS,
    _LTX2TPAttnProcessor,
    _difflet_apply_split_rotary_emb,
    _shard_ltx2_transformer,
)
from difflet.ops import attention as difflet_attention


def build_ltx2_transformer(config):
    """diffusers' ``LTX2VideoTransformer3DModel`` from a difflet config object.

    Installs difflet's trace-safe split RoPE into the diffusers module first
    (see ``_difflet_apply_split_rotary_emb``).
    """
    import diffusers.models.transformers.transformer_ltx2 as ltx2_transformer

    ltx2_transformer.apply_split_rotary_emb = _difflet_apply_split_rotary_emb
    LTX2VideoTransformer3DModel = ltx2_transformer.LTX2VideoTransformer3DModel
    return LTX2VideoTransformer3DModel(
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


def _to_bh(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """[B, S, H*D] -> [B*H, S, D], the tp_q layout difflet's attention op takes."""
    bsz, length, inner = x.shape
    head_dim = inner // n_heads
    return x.reshape(bsz, length, n_heads, head_dim).permute(0, 2, 1, 3).reshape(
        bsz * n_heads, length, head_dim
    )


def _key_window(attention_mask: torch.Tensor, rows: int, q_len: int, kv_len: int) -> dict:
    """``bound_min`` / ``bound_max`` for an additive key-padding mask.

    The transformer hands the processor ``(1 - m) * -10000`` viewed as ``(B, H, 1, kv)``;
    the valid keys are one contiguous run (left- or right-padded tokenizer), so they are
    exactly a ``[bound_min, bound_max)`` window per row. Device-side argmin / argmax, no
    host sync.
    """
    valid = attention_mask.reshape(rows, -1, kv_len)[:, :1, :] > -1.0
    idx = torch.arange(kv_len, device=valid.device, dtype=torch.int32).view(1, 1, -1)
    lo = torch.where(valid, idx, torch.full_like(idx, kv_len)).amin(dim=-1, keepdim=True)
    hi = torch.where(valid, idx + 1, torch.zeros_like(idx)).amax(dim=-1, keepdim=True)
    return {
        "bound_min": lo.expand(rows, q_len, 1).contiguous(),
        "bound_max": hi.expand(rows, q_len, 1).contiguous(),
    }


class TpuLTX2AttnProcessor(_LTX2TPAttnProcessor):
    """The shared TP processor, with cross-attention masked by key windows.

    Self-attention (no mask) is the shared processor's path, unchanged. With a mask the
    projection, global qk RMS norm and RoPE are the shared ones too; only the attention
    call differs: it gets the mask's key window instead of attending over text padding.
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
    ):
        if attention_mask is None:
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                query_rotary_emb, key_rotary_emb,
            )
        import diffusers.models.transformers.transformer_ltx2 as ltx2_transformer

        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch_size, sequence_length, _ = context.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = self._global_rms_norm(attn.norm_q, attn.to_q(hidden_states))
        key = self._global_rms_norm(attn.norm_k, attn.to_k(context))
        value = attn.to_v(context)
        if query_rotary_emb is not None:
            k_rope = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            if attn.rope_type == "interleaved":
                query = ltx2_transformer.apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = ltx2_transformer.apply_interleaved_rotary_emb(key, k_rope)
            elif attn.rope_type == "split":
                query = ltx2_transformer.apply_split_rotary_emb(query, query_rotary_emb)
                key = ltx2_transformer.apply_split_rotary_emb(key, k_rope)

        out_dtype = query.dtype
        n_heads = attn.heads
        bsz, q_len, inner = query.shape
        kv_len = key.shape[1]
        head_dim = inner // n_heads
        out3 = difflet_attention(
            _to_bh(query, n_heads), _to_bh(key, n_heads), _to_bh(value, n_heads),
            scale=1.0 / math.sqrt(head_dim),
            causal=False,
            tp_q=True, tp_k=True, tp_out=False,
            **_key_window(attention_mask, bsz * n_heads, q_len, kv_len),
        )
        hidden_states = out3.reshape(bsz, n_heads, q_len, head_dim).permute(0, 2, 1, 3).reshape(
            bsz, q_len, inner
        )
        hidden_states = attn.to_out[0](hidden_states.to(out_dtype))
        return attn.to_out[1](hidden_states)


def shard_ltx2_transformer(transformer: nn.Module, tp_degree: int, rank_util) -> None:
    """The shared TP sharding, then the TPU processor on every sharded attention."""
    _shard_ltx2_transformer(transformer, tp_degree, rank_util)
    if tp_degree <= 1:
        return
    processor = TpuLTX2AttnProcessor(tp_degree=tp_degree, rank_util=rank_util)
    for block in transformer.transformer_blocks:
        for name in _LTX2_ATTN_ATTRS:
            attn = getattr(block, name)
            if isinstance(attn.processor, _LTX2TPAttnProcessor):
                attn.processor = processor
