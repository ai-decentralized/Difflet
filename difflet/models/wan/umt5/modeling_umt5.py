"""UMT5 encoder modeling for Wan 2.2 — hardware-agnostic.

Imports only ``torch``, stdlib, ``transformers`` activation registry, and
``difflet.ops`` — never ``neuronx_distributed`` / ``nkilib`` / ``torch_neuronx``
directly. State-dict keys mirror upstream ``UMT5EncoderModel`` so a HF
checkpoint loads with no key renaming:

```text
shared.weight
encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight   # per-layer (UMT5)
encoder.block.{i}.layer.0.layer_norm.weight
encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight
encoder.block.{i}.layer.1.layer_norm.weight
encoder.final_layer_norm.weight
```

Key UMT5-vs-T5 differences captured here:

- Per-layer relative attention bias (every block has its own bias embedding,
  not just block 0).
- RMSNorm without mean centering (same as T5).
- Gated FFN by default (`is_gated_act=True`, `feed_forward_proj="gated-gelu"`)
  using a parallel ``wi_0`` / ``wi_1`` pair fused via element-wise product.

Trainium TP layout decisions (mirror flux T5):

- ``shared`` (token embedding): ``ParallelEmbedding(shard_across_embedding=True)``.
- Attention q/k/v/o: ``ColumnParallelLinear(gather_output=True)`` — replicated
  attention compute; T5 inner_dim is small (4096) so the cost is acceptable.
- FFN ``wi_0`` / ``wi_1``: ``ColumnParallelLinear(gather_output=False)`` →
  GELU-gate → ``RowParallelLinear(input_is_parallel=True)`` for ``wo``.
- Per-layer relative position bias embedding:
  ``ParallelEmbedding(shard_across_embedding=True)`` — small (32 × 64) but
  matches the flux T5 pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from difflet.ops import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RMSNorm,
    RowParallelLinear,
)


@dataclass
class WanUmT5Config:
    """Hyperparameters for ``WanUmT5EncoderModel``.

    Defaults match the umt5-xxl variant shipped with
    ``Wan-AI/Wan2.2-T2V-A14B-Diffusers/text_encoder/config.json``.
    """

    vocab_size: int = 256384
    d_model: int = 4096
    d_kv: int = 64
    d_ff: int = 10240
    num_heads: int = 64
    num_layers: int = 24
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    is_gated_act: bool = True
    dense_act_fn: str = "gelu_new"
    feed_forward_proj: str = "gated-gelu"
    layer_norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not self.is_gated_act:
            raise NotImplementedError(
                "Wan UMT5 spike currently supports only gated FFN "
                "(is_gated_act=True). Wan2.2 UMT5-XXL ships gated-gelu."
            )

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.d_kv

    @classmethod
    def from_diffusers_dict(cls, raw: dict) -> "WanUmT5Config":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kept = {k: v for k, v in raw.items() if k in fields}
        return cls(**kept)


# ---------------------------------------------------------------------------
# Feed-forward (gated GELU)


class WanUmT5DenseGatedActDense(nn.Module):
    """Two-input gated FFN: ``act(wi_0(x)) * wi_1(x) -> wo(...)``.

    Matches HF UMT5DenseGatedActDense submodule names so checkpoints load
    without renaming.
    """

    def __init__(self, config: WanUmT5Config):
        super().__init__()
        self.wi_0 = ColumnParallelLinear(
            config.d_model, config.d_ff, bias=False, gather_output=False
        )
        self.wi_1 = ColumnParallelLinear(
            config.d_model, config.d_ff, bias=False, gather_output=False
        )
        self.wo = RowParallelLinear(
            config.d_ff, config.d_model, bias=False, input_is_parallel=True
        )
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.wo(hidden_states)
        return hidden_states


class WanUmT5LayerFF(nn.Module):
    """Pre-norm residual wrapper around ``WanUmT5DenseGatedActDense``."""

    def __init__(self, config: WanUmT5Config):
        super().__init__()
        self.DenseReluDense = WanUmT5DenseGatedActDense(config)
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states


# ---------------------------------------------------------------------------
# Attention with relative position bias


class WanUmT5Attention(nn.Module):
    """UMT5 self-attention with per-layer relative position bias.

    Q/K/V/O are all replicated across TP (``gather_output=True``) — same
    pattern as flux's ``NeuronT5Attention``. The relative position bias
    embedding is sharded across heads via ``ParallelEmbedding``; we gather
    the per-head bias into a full ``(1, num_heads, S_q, S_k)`` tensor inside
    ``compute_bias`` because the rest of attention runs replicated.

    No 1/sqrt(d_kv) scaling at runtime — T5/UMT5 absorb that into the q/k
    initialization so the forward pass is bias-only.
    """

    def __init__(self, config: WanUmT5Config, has_relative_attention_bias: bool = True):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = ColumnParallelLinear(
            self.d_model, self.inner_dim, bias=False, gather_output=True
        )
        self.k = ColumnParallelLinear(
            self.d_model, self.inner_dim, bias=False, gather_output=True
        )
        self.v = ColumnParallelLinear(
            self.d_model, self.inner_dim, bias=False, gather_output=True
        )
        self.o = ColumnParallelLinear(
            self.inner_dim, self.d_model, bias=False, gather_output=True
        )

        if has_relative_attention_bias:
            self.relative_attention_bias = ParallelEmbedding(
                self.relative_attention_num_buckets,
                self.n_heads,
                shard_across_embedding=True,
                pad=False,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(
        self, query_length: int, key_length: int, device=None
    ) -> torch.Tensor:
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,  # encoder-only
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        # (Q, K, num_heads) -> (1, num_heads, Q, K)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states: torch.Tensor) -> torch.Tensor:
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states: torch.Tensor) -> torch.Tensor:
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        query_states = shape(self.q(hidden_states))  # (B, H, S, d_kv)
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))

        scores = torch.matmul(query_states, key_states.transpose(3, 2))  # (B, H, S_q, S_k)

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(seq_length, seq_length, device=scores.device)
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, seq_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask

        scores = scores + position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)
        return attn_output, position_bias


class WanUmT5LayerSelfAttention(nn.Module):
    """Pre-norm residual wrapper around ``WanUmT5Attention``."""

    def __init__(self, config: WanUmT5Config, has_relative_attention_bias: bool = True):
        super().__init__()
        self.SelfAttention = WanUmT5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed_hidden_states, mask=attention_mask, position_bias=position_bias
        )
        hidden_states = hidden_states + attn_output
        return hidden_states, position_bias


class WanUmT5Block(nn.Module):
    """One UMT5 encoder block: self-attention + feed-forward.

    UMT5 uses per-layer relative bias, so every block instantiates its own
    bias embedding (unlike T5 where only block 0 owns it).
    """

    def __init__(self, config: WanUmT5Config):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(
            WanUmT5LayerSelfAttention(config, has_relative_attention_bias=True)
        )
        self.layer.append(WanUmT5LayerFF(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, position_bias = self.layer[0](
            hidden_states, attention_mask=attention_mask, position_bias=position_bias
        )
        hidden_states = self.layer[-1](hidden_states)
        return hidden_states, position_bias


class WanUmT5Stack(nn.Module):
    """UMT5 encoder stack: embedding + N blocks + final layer norm."""

    def __init__(self, config: WanUmT5Config, embed_tokens: nn.Module):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.block = nn.ModuleList(
            [WanUmT5Block(config) for _ in range(config.num_layers)]
        )
        self.final_layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None:
            # Convert (B, S) padding mask to additive (B, 1, 1, S) form.
            extended = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
            extended = (1.0 - extended) * torch.finfo(inputs_embeds.dtype).min
        else:
            extended = None

        hidden_states = inputs_embeds
        # UMT5: each block computes its own position_bias, never reuses across layers.
        for block in self.block:
            hidden_states, _ = block(
                hidden_states, attention_mask=extended, position_bias=None
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class WanUmT5EncoderModel(nn.Module):
    """Top-level UMT5 encoder.

    forward args:
      input_ids: (B, S) int64 token IDs.
      attention_mask: optional (B, S) {0,1} padding mask.

    returns:
      last_hidden_state: (B, S, d_model) text embeddings ready for Wan DiT
      cross-attention.
    """

    def __init__(self, config: WanUmT5Config):
        super().__init__()
        self.config = config
        self.shared = ParallelEmbedding(
            config.vocab_size,
            config.d_model,
            shard_across_embedding=True,
            pad=False,
        )
        self.encoder = WanUmT5Stack(config, embed_tokens=self.shared)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(input_ids, attention_mask=attention_mask)


__all__ = [
    "WanUmT5Attention",
    "WanUmT5Block",
    "WanUmT5Config",
    "WanUmT5DenseGatedActDense",
    "WanUmT5EncoderModel",
    "WanUmT5LayerFF",
    "WanUmT5LayerSelfAttention",
    "WanUmT5Stack",
]
