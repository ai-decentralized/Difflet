"""Backend-neutral tensor-parallel rewrite of the MiniMax-H3 Omni Transformer.

Shared by the Trainium AoT backend (``difflet/backends/trainium/minimax_h3/transformer.py``) and
the PyTorch-native neuron backend (``difflet/backends/neuron/minimax_h3/transformer.py``). Every
layer here comes from ``difflet.ops``, so importing this module binds ``difflet.ops`` to the
selected backend. ``difflet.ops`` is therefore imported inside the functions that build layers
(after the backend's mesh exists), never at module import.

* ``_replace_h3_linears_for_tp`` head-shards every attention (56 heads / tp), splits each fused
  SwiGLU into column-parallel up/gate and a row-parallel down projection, and column-shards the
  text embedder, the output norm's modulation and (when present) the 13B ``adaln_proj``.
* ``MiniMaxH3TPAttnProcessor`` runs the self-attention through ``difflet.ops.attention`` with the
  packed sequence's padded tail lowered to per-query key bounds.
* ``_MiniMaxH3TransformerTraceModule`` builds the diffusers-layout model from any config object
  carrying the transformer's hyper-parameters and applies the TP rewrite.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class _H3TensorParallelSwiGLU(nn.Module):
    """Shard both SwiGLU branches without mixing their checkpoint row layout."""

    def __init__(self, hidden_size: int, ffn_dim: int) -> None:
        from difflet.ops import ColumnParallelLinear, RowParallelLinear

        super().__init__()
        self.up_proj = ColumnParallelLinear(hidden_size, ffn_dim, bias=False, gather_output=False)
        self.gate_proj = ColumnParallelLinear(hidden_size, ffn_dim, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(ffn_dim, hidden_size, bias=False, input_is_parallel=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.up_proj(hidden_states) * torch.nn.functional.silu(self.gate_proj(hidden_states))
        )


def _column_parallel_like(linear: nn.Linear, *, gather_output: bool) -> nn.Module:
    from difflet.ops import ColumnParallelLinear

    return ColumnParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        gather_output=gather_output,
    )


def _row_parallel_like(linear: nn.Linear) -> nn.Module:
    from difflet.ops import RowParallelLinear

    return RowParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        input_is_parallel=True,
    )


def _safe_tp_size() -> int:
    from difflet.ops import get_tensor_model_parallel_size

    try:
        return int(get_tensor_model_parallel_size())
    except AssertionError:
        return 1


def _replace_attention_for_tp(attn, tp_degree: int) -> None:
    if int(attn.heads) % tp_degree:
        raise ValueError(f"MiniMax-H3 attention heads {attn.heads} must divide TP{tp_degree}.")
    attn.heads = int(attn.heads) // tp_degree
    attn.to_q = _column_parallel_like(attn.to_q, gather_output=False)
    attn.to_k = _column_parallel_like(attn.to_k, gather_output=False)
    attn.to_v = _column_parallel_like(attn.to_v, gather_output=False)
    attn.to_out[0] = _row_parallel_like(attn.to_out[0])
    attn.processor = MiniMaxH3TPAttnProcessor()


def _replace_block_for_tp(block, tp_degree: int, hidden_size: int, ffn_dim: int) -> None:
    _replace_attention_for_tp(block.attn, tp_degree)
    block.ff = _H3TensorParallelSwiGLU(hidden_size, ffn_dim)


def _replace_h3_linears_for_tp(transformer: nn.Module) -> None:
    tp_degree = _safe_tp_size()
    if tp_degree <= 1:
        return

    hidden_size = int(transformer.config.hidden_size)
    ffn_dim = int(transformer.config.ffn_dim)
    transformer.context_embedder = _column_parallel_like(
        transformer.context_embedder, gather_output=True
    )
    transformer.norm_out.linear = _column_parallel_like(
        transformer.norm_out.linear, gather_output=True
    )
    for block in transformer.token_refiner.refiner_blocks:
        _replace_block_for_tp(block, tp_degree, hidden_size, ffn_dim)
    for block in transformer.transformer_blocks:
        _replace_block_for_tp(block, tp_degree, hidden_size, ffn_dim)
        # The 13B AdaLN branch is sharded in HBM; its tiny per-step output is
        # gathered because the residual stream remains replicated after TP.
        # Absent entirely when the modulation table is precomputed on the host.
        if block.adaln_proj is not None:
            block.adaln_proj.linear = _column_parallel_like(
                block.adaln_proj.linear, gather_output=True
            )


def _prefix_bounds(
    attention_mask: torch.Tensor,
    *,
    num_heads: int,
    query_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact prefix-validity mask to attention_cte per-query bounds."""

    valid = attention_mask if attention_mask.dtype == torch.bool else attention_mask != 0
    count = valid.sum(dim=-1, keepdim=True).to(torch.int32)
    batch_size = int(valid.shape[0])
    bound_max = (
        count[:, None, None, :]
        .expand(batch_size, num_heads, query_length, 1)
        .reshape(batch_size * num_heads, query_length, 1)
        .contiguous()
    )
    return torch.zeros_like(bound_max), bound_max


class MiniMaxH3TPAttnProcessor:
    """Full H3 self-attention through attention_cte, including tail bounds."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self) -> None:
        # Bound once here, not imported inside the (compiled) call.
        from difflet.models.minimax_h3.modeling_minimax_h3 import _apply_rotary_emb
        from difflet.ops import attention

        self._attention = attention
        self._apply_rotary_emb = _apply_rotary_emb

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _apply_rotary_emb = self._apply_rotary_emb
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        batch_size, num_heads, query_length, head_dim = q.shape
        key_length = int(k.shape[2])
        kwargs = {}
        if attention_mask is not None:
            bound_min, bound_max = _prefix_bounds(
                attention_mask,
                num_heads=num_heads,
                query_length=query_length,
            )
            kwargs.update(bound_min=bound_min, bound_max=bound_max)
        output = self._attention(
            q.reshape(batch_size * num_heads, query_length, head_dim),
            k.reshape(batch_size * num_heads, key_length, head_dim),
            v.reshape(batch_size * num_heads, key_length, head_dim),
            scale=1.0 / math.sqrt(head_dim),
            causal=False,
            tp_q=True,
            tp_k=True,
            tp_out=False,
            **kwargs,
        )
        output = output.reshape(batch_size, num_heads, query_length, head_dim)
        output = output.transpose(1, 2).flatten(2, 3).to(query.dtype)
        output = attn.to_out[0](output.contiguous())
        return attn.to_out[1](output)


class _MiniMaxH3TransformerTraceModule(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        from difflet.models.minimax_h3.modeling_minimax_h3 import (
            MiniMaxH3Transformer3DModel,
        )

        self.config = config
        self.transformer = MiniMaxH3Transformer3DModel(
            num_attention_heads=int(config.num_attention_heads),
            attention_head_dim=int(config.attention_head_dim),
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_layers),
            num_refiner_layers=int(config.num_refiner_layers),
            ffn_dim=int(config.ffn_dim),
            in_channels=int(config.in_channels),
            audio_in_channels=int(config.audio_in_channels),
            patch_size=tuple(config.patch_size),
            text_dim=int(config.text_dim),
            freq_dim=int(config.freq_dim),
            time_embed_hidden_dim=int(config.time_embed_hidden_dim),
            time_embed_dim=int(config.time_embed_dim),
            rope_freq_dim=int(config.rope_freq_dim),
            rope_theta=float(config.rope_theta),
            norm_eps=float(config.norm_eps),
            qk_norm_eps=float(config.qk_norm_eps),
            final_norm_eps=float(config.final_norm_eps),
            precomputed_adaln=bool(config.precomputed_adaln),
        )
        _replace_h3_linears_for_tp(self.transformer)

    def restore_checkpoint_fp32_modules(self) -> None:
        transformer = self.transformer
        transformer.proj_in.float()
        transformer.audio_proj_in.float()
        transformer.time_embedder.float()
        transformer.proj_out.float()
        transformer.audio_proj_out.float()

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        encoder_attention_mask,
        attention_mask,
        block_modulation=None,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=attention_mask,
            block_modulation=block_modulation,
            return_dict=False,
        )


# The Trainium module's historical name for the processor.
_MiniMaxH3TrainiumAttnProcessor = MiniMaxH3TPAttnProcessor


def split_fused_swiglu_keys(state_dict: dict, prefixes) -> dict:
    """Rename each ``{prefix}.net.0.proj.weight`` (diffusers' fused SwiGLU, rows ``[up; gate]``)
    and ``{prefix}.net.2.weight`` to the ``up_proj``/``gate_proj``/``down_proj`` the TP rewrite
    uses. In place; returns ``state_dict``."""
    for prefix in prefixes:
        fused_key = f"{prefix}.net.0.proj.weight"
        if fused_key not in state_dict:
            continue
        up_weight, gate_weight = state_dict.pop(fused_key).chunk(2, dim=0)
        state_dict[f"{prefix}.up_proj.weight"] = up_weight.contiguous()
        state_dict[f"{prefix}.gate_proj.weight"] = gate_weight.contiguous()
        state_dict[f"{prefix}.down_proj.weight"] = state_dict.pop(f"{prefix}.net.2.weight")
    return state_dict


def ff_prefixes(num_refiner_layers: int, num_layers: int, root: str = "transformer.") -> list[str]:
    """The feed-forward module prefixes of every refiner block and transformer block."""
    return [
        *[f"{root}token_refiner.refiner_blocks.{i}.ff" for i in range(int(num_refiner_layers))],
        *[f"{root}transformer_blocks.{i}.ff" for i in range(int(num_layers))],
    ]
