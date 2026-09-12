"""Tensor-parallel sharding of diffusers' ``FluxTransformer2DModel``.

Backend-neutral (everything goes through ``difflet.ops``), written for the TPU
port of FLUX.1-dev — the Trainium Flux path is the legacy NxDI fork in
``modeling_flux.py`` and is left untouched (option (b) of
docs/plans/2026-09-11-tpu-port-hunyuan-ltx2-flux.md). Same shape as the LTX-2
recipe (``models/ltx_2/tp_sharding.py``):

* joint blocks: ``to_q/k/v`` and ``add_q/k/v_proj`` column-parallel
  (gather_output=False), ``to_out[0]`` / ``to_add_out`` row-parallel, both
  FeedForwards column→row;
* single blocks: ``to_q/k/v`` and ``proj_mlp`` column-parallel; the fused
  ``proj_out`` over ``cat([attn, mlp])`` becomes two row-parallel halves
  (``proj_out_attn`` over the head-sharded attention output, ``proj_out_mlp``
  over the column-sharded MLP), summed and all-reduced once — the same split
  HunyuanVideo's single blocks use, because a fused row-parallel over a
  concatenation pairs each rank's ``[attn_r || mlp_r]`` with the wrong weight
  columns;
* the qk RMSNorm is per head (``RMSNorm(dim_head)``), so head sharding keeps it
  local; RoPE is per position and shared by all heads, so nothing to slice.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from difflet.ops import (
    ColumnParallelLinear,
    RowParallelLinear,
    attention as difflet_attention,
    get_tensor_model_parallel_size,
    reduce_from_tensor_model_parallel_region,
)

FLUX_ATTN_COLUMN = ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")


def build_flux_transformer(config: Any):
    """diffusers' ``FluxTransformer2DModel`` from a difflet config object."""
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

    return FluxTransformer2DModel(
        patch_size=int(getattr(config, "patch_size", 1)),
        in_channels=int(config.in_channels),
        out_channels=getattr(config, "out_channels", None),
        num_layers=int(config.num_layers),
        num_single_layers=int(config.num_single_layers),
        attention_head_dim=int(config.attention_head_dim),
        num_attention_heads=int(config.num_attention_heads),
        joint_attention_dim=int(config.joint_attention_dim),
        pooled_projection_dim=int(config.pooled_projection_dim),
        guidance_embeds=bool(getattr(config, "guidance_embeds", True)),
        axes_dims_rope=tuple(int(v) for v in getattr(config, "axes_dims_rope", (16, 56, 56))),
    )


def safe_tensor_parallel_size() -> int:
    try:
        return int(get_tensor_model_parallel_size())
    except (AssertionError, RuntimeError):
        return 1


def column_parallel_like(linear: nn.Linear) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        linear.in_features, linear.out_features, bias=linear.bias is not None, gather_output=False
    )


def row_parallel_like(linear: nn.Linear, **kwargs) -> RowParallelLinear:
    return RowParallelLinear(
        linear.in_features, linear.out_features, bias=linear.bias is not None,
        input_is_parallel=True, **kwargs,
    )


class FluxTPAttnProcessor:
    """``FluxAttnProcessor`` over this rank's heads, through ``difflet.ops.attention``.

    Faithful to the stock processor: per-head qk RMSNorm, text tokens first in
    the joint sequence, shared RoPE, no mask; only the attention call differs
    (heads folded into the batch axis, ``tp_q/tp_k`` so the op knows the heads
    are already sharded).
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        heads = int(attn.heads)
        query = query.unflatten(-1, (heads, -1))
        key = key.unflatten(-1, (heads, -1))
        value = value.unflatten(-1, (heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if attn.added_kv_proj_dim is not None:
            eq = attn.norm_added_q(attn.add_q_proj(encoder_hidden_states).unflatten(-1, (heads, -1)))
            ek = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states).unflatten(-1, (heads, -1)))
            ev = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (heads, -1))
            query = torch.cat([eq, query], dim=1)
            key = torch.cat([ek, key], dim=1)
            value = torch.cat([ev, value], dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        bsz, seq, _, head_dim = query.shape
        out_dtype = query.dtype
        q3 = query.permute(0, 2, 1, 3).reshape(bsz * heads, seq, head_dim)
        k3 = key.permute(0, 2, 1, 3).reshape(bsz * heads, seq, head_dim)
        v3 = value.permute(0, 2, 1, 3).reshape(bsz * heads, seq, head_dim)
        if attention_mask is not None:
            raise NotImplementedError("Flux TP attention does not take an attention_mask")
        out3 = difflet_attention(
            q3, k3, v3, scale=1.0 / math.sqrt(head_dim), causal=False,
            tp_q=True, tp_k=True, tp_out=False,
        )
        out = out3.reshape(bsz, heads, seq, head_dim).permute(0, 2, 1, 3).reshape(bsz, seq, heads * head_dim)
        out = out.to(out_dtype)
        if encoder_hidden_states is not None:
            text_len = encoder_hidden_states.shape[1]
            enc, img = out[:, :text_len], out[:, text_len:]
            img = attn.to_out[0](img.contiguous())
            img = attn.to_out[1](img)
            enc = attn.to_add_out(enc.contiguous())
            return img, enc
        return out


def _single_block_forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                          joint_attention_kwargs=None):
    """``FluxSingleTransformerBlock.forward`` with the split, once-reduced proj_out."""
    text_seq_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    attn_output = self.attn(
        hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb,
        **(joint_attention_kwargs or {}),
    )
    res_attn = self.proj_out_attn(attn_output)
    out_attn, attn_bias = res_attn if isinstance(res_attn, tuple) else (res_attn, None)
    out_mlp = self.proj_out_mlp(mlp_hidden_states)
    proj = reduce_from_tensor_model_parallel_region(out_attn + out_mlp)
    if attn_bias is not None:
        proj = proj + attn_bias
    hidden_states = residual + gate.unsqueeze(1) * proj
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]


def shard_flux_transformer(transformer: nn.Module, tp_degree: int) -> None:
    """Head-shard every block's attention and column/row-shard the FFNs in place."""
    if tp_degree <= 1:
        return
    processor = FluxTPAttnProcessor()
    for block in transformer.transformer_blocks:
        attn = block.attn
        if int(attn.heads) % int(tp_degree):
            raise ValueError(f"Flux heads {attn.heads} must divide tp={tp_degree}")
        attn.heads = int(attn.heads) // int(tp_degree)
        attn.inner_dim = int(attn.inner_dim) // int(tp_degree)
        for name in FLUX_ATTN_COLUMN:
            setattr(attn, name, column_parallel_like(getattr(attn, name)))
        attn.to_out[0] = row_parallel_like(attn.to_out[0])
        attn.to_add_out = row_parallel_like(attn.to_add_out)
        attn.processor = processor
        block.ff.net[0].proj = column_parallel_like(block.ff.net[0].proj)
        block.ff.net[2] = row_parallel_like(block.ff.net[2])
        block.ff_context.net[0].proj = column_parallel_like(block.ff_context.net[0].proj)
        block.ff_context.net[2] = row_parallel_like(block.ff_context.net[2])
    for block in transformer.single_transformer_blocks:
        attn = block.attn
        attn.heads = int(attn.heads) // int(tp_degree)
        attn.inner_dim = int(attn.inner_dim) // int(tp_degree)
        for name in ("to_q", "to_k", "to_v"):
            setattr(attn, name, column_parallel_like(getattr(attn, name)))
        attn.processor = processor
        dim = block.proj_out.out_features
        mlp_dim = int(block.mlp_hidden_dim)
        block.proj_mlp = column_parallel_like(block.proj_mlp)
        block.proj_out_attn = RowParallelLinear(
            dim, dim, bias=True, input_is_parallel=True, reduce_output=False, skip_bias_add=True
        )
        block.proj_out_mlp = RowParallelLinear(
            mlp_dim, dim, bias=False, input_is_parallel=True, reduce_output=False
        )
        del block.proj_out
        block.forward = _single_block_forward.__get__(block, type(block))


__all__ = [
    "FLUX_ATTN_COLUMN",
    "FluxTPAttnProcessor",
    "build_flux_transformer",
    "column_parallel_like",
    "row_parallel_like",
    "safe_tensor_parallel_size",
    "shard_flux_transformer",
]
