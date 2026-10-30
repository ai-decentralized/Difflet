"""Parallel building blocks for the HunyuanVideo 1.5 AOT DiT (traced NxD SPMD graph).

* ``hv15_text_order``: host-side valid-first permutation of the joint text stream.
* ``install_tensor_parallel``: swaps the dual-stream blocks' linears for difflet.ops
  column/row-parallel layers in place (parameter names unchanged).
* ``bounded_joint_attention`` / ``JointBoundedProcessor``: joint image+text attention
  over the contiguous valid-key prefix ``[0, N + text_valid)`` via attention_cte's
  bound range, with the sequence padded in-graph to a multiple of 128.

Only difflet.ops (backend-dispatched) and HV1.0's ``_sp_unbias`` are used here, so the
same code runs the CPU reference and the Trainium graph.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.models.hunyuan_video.modeling_hunyuan_video import _sp_unbias
from difflet.ops import (
    ColumnParallelLinear,
    RowParallelLinear,
    apply_rotary_emb,
    attention,
    gather_from_sequence_parallel_region,
    get_tensor_model_parallel_size,
    reduce_scatter_to_sequence_parallel_region,
)

_ATTENTION_TILE = 128


def hv15_text_order(
    mask_image: torch.Tensor, mask_byt5: torch.Tensor, mask_mllm: torch.Tensor
) -> torch.LongTensor:
    """Host-side valid-first order of the joint text stream ``[image_3, byt5, mllm]``.

    Stable, so valid tokens keep their order and come first, exactly diffusers'
    ``[valid_image, valid_byt5, valid_mllm, invalid_image, invalid_byt5, invalid_mllm]``
    reorder. ``mask_image`` follows the static forward: all 0 for t2v, all 1 otherwise.
    """
    mask = torch.cat([mask_image, mask_byt5, mask_mllm], dim=1).bool()
    return torch.argsort((~mask).to(torch.int8), dim=1, stable=True)


def bounded_joint_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_valid_len: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Attention of every query over keys ``[0, key_valid_len)``; ``[B, H, S, d]`` in/out.

    q, k and v are padded to a multiple of 128 with a static ``F.pad``; real query rows
    attend ``[0, key_valid_len)`` and padded query rows ``[0, 1)`` (finite, sliced off).
    Bounds use expand/reshape/cat only (trace-safe; validated on device by the Task 0
    spike). ``key_valid_len`` is an int32 ``[B]`` count accumulated in fp32 upstream.
    """
    batch, heads, seq, head_dim = q.shape
    pad = -seq % _ATTENTION_TILE
    padded = seq + pad
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    bound_max = key_valid_len.to(torch.int32).reshape(batch, 1, 1, 1).expand(batch, heads, seq, 1)
    if pad:
        pad_rows = torch.ones((batch, heads, pad, 1), dtype=torch.int32, device=q.device)
        bound_max = torch.cat([bound_max, pad_rows], dim=2)
    bound_max = bound_max.reshape(batch * heads, padded, 1).contiguous()
    out = attention(
        q.reshape(batch * heads, padded, head_dim),
        k.reshape(batch * heads, padded, head_dim),
        v.reshape(batch * heads, padded, head_dim),
        scale=scale,
        causal=False,
        bound_min=torch.zeros_like(bound_max),
        bound_max=bound_max,
        tp_q=True,
        tp_k=True,
        tp_out=False,
    )
    return out.reshape(batch, heads, padded, head_dim)[:, :, :seq]


def _rotary(x: torch.Tensor, image_rotary_emb) -> torch.Tensor:
    # diffusers' apply_rotary_emb(use_real, sequence_dim=1) computes in fp32.
    cos, sin = (t[None, :, None, :] for t in image_rotary_emb)
    return apply_rotary_emb(x.float(), cos, sin).to(x.dtype)


class JointBoundedProcessor:
    """HunyuanVideo15AttnProcessor2_0 with key-bounded attention instead of a dense mask.

    ``attention_mask`` carries ``text_valid`` (int32 ``[B]``): the static forward has
    already put the valid text tokens first, so the valid keys are the contiguous prefix
    ``[0, N + text_valid)`` of ``[image ‖ text]``. Under Megatron-SP the image stream
    arrives sequence-sharded and is gathered first; its out-projection is reduce-scattered.
    """

    def __init__(self, sp: bool = False) -> None:
        self.sp = sp

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ):
        if self.sp:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, dim=1)
        num_image_tokens = hidden_states.shape[1]

        query = attn.norm_q(attn.to_q(hidden_states).unflatten(2, (attn.heads, -1)))
        key = attn.norm_k(attn.to_k(hidden_states).unflatten(2, (attn.heads, -1)))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        if image_rotary_emb is not None:
            query = _rotary(query, image_rotary_emb)
            key = _rotary(key, image_rotary_emb)

        encoder_query = attn.add_q_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        encoder_key = attn.add_k_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)

        out = bounded_joint_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            num_image_tokens + attention_mask,
            scale=1.0 / math.sqrt(query.shape[-1]),
        )
        out = out.transpose(1, 2).flatten(2, 3).to(query.dtype)

        hidden_states = attn.to_out[0](out[:, :num_image_tokens])
        if self.sp:
            hidden_states = reduce_scatter_to_sequence_parallel_region(hidden_states, dim=1)
            hidden_states = _sp_unbias(hidden_states, attn.to_out[0])
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(out[:, num_image_tokens:])
        return hidden_states, encoder_hidden_states


class _SequenceParallelFeedForward(nn.Module):
    """Image-stream FFN under Megatron-SP: gather (g) -> net -> reduce-scatter (ḡ).

    Holds the diffusers FeedForward's ``net`` so parameter names stay ``ff.net.*``.
    """

    def __init__(self, net: nn.ModuleList) -> None:
        super().__init__()
        self.net = net

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = gather_from_sequence_parallel_region(hidden_states, dim=1)
        for module in self.net:
            hidden_states = module(hidden_states)
        hidden_states = reduce_scatter_to_sequence_parallel_region(hidden_states, dim=1)
        return _sp_unbias(hidden_states, self.net[2])


def _copy_weights(new: nn.Module, old: nn.Linear) -> nn.Module:
    # Same shape only at tp == 1 (CPU / unsharded); sharded ranks load from the checkpoint.
    if new.weight.shape == old.weight.shape:
        with torch.no_grad():
            new.weight.copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
    return new


def _column(linear: nn.Linear, *, gather_output: bool) -> nn.Module:
    layer = ColumnParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        gather_output=gather_output,
    )
    return _copy_weights(layer, linear)


def _row(linear: nn.Linear, *, reduce_output: bool) -> nn.Module:
    layer = RowParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        input_is_parallel=True,
        reduce_output=reduce_output,
    )
    return _copy_weights(layer, linear)


def install_tensor_parallel(model: nn.Module, sp: bool = False) -> None:
    """Shard the dual-stream ``transformer_blocks`` of a diffusers HV1.5 model in place.

    q/k/v + add_q/k/v column-parallel, ``to_out[0]`` / ``to_add_out`` row-parallel, MLPs
    column then row, AdaLN ``norm1(_context).linear`` column-parallel with gathered
    output. Heads per rank come from the runtime TP size (1 on CPU), as in HV1.0. The
    token refiner runs on host and is left alone. Under SP only the image stream is
    sequence-sharded: its row-parallel outputs skip the all-reduce (folded into ḡ).
    """
    tp = get_tensor_model_parallel_size()
    for block in model.transformer_blocks:
        attn = block.attn
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            setattr(attn, name, _column(getattr(attn, name), gather_output=False))
        attn.to_out[0] = _row(attn.to_out[0], reduce_output=not sp)
        attn.to_add_out = _row(attn.to_add_out, reduce_output=True)
        attn.heads //= tp

        for ff, seq_sharded in ((block.ff, sp), (block.ff_context, False)):
            ff.net[0].proj = _column(ff.net[0].proj, gather_output=False)
            ff.net[2] = _row(ff.net[2], reduce_output=not seq_sharded)
        if sp:
            block.ff = _SequenceParallelFeedForward(block.ff.net)

        block.norm1.linear = _column(block.norm1.linear, gather_output=True)
        block.norm1_context.linear = _column(block.norm1_context.linear, gather_output=True)


__all__ = [
    "JointBoundedProcessor",
    "bounded_joint_attention",
    "hv15_text_order",
    "install_tensor_parallel",
]
