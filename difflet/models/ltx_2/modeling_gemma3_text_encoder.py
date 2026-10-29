"""Gemma-3's language model, text-only, as LTX-2's prompt encoder. Hardware-agnostic.

LTX-2 conditions on every hidden state of Gemma-3 12B's language model (diffusers
``LTX2Pipeline._get_gemma_prompt_embeds``: ``output_hidden_states=True``, the 49 states stacked
on a new last axis and flattened into ``[B, L, 3840 * 49]``), which its text connectors then turn
into the DiT's two text streams. A prompt is text only, so the SigLIP vision tower, the
multimodal projector and the ``lm_head`` never run; this module is the decoder stack, with the TP
layers of ``difflet.ops`` so the neuron backend can shard its 11.8 B parameters (21.9 GiB in
bf16) over the TP group. Imports only ``torch``, stdlib and ``difflet.ops``.

The module mirrors transformers' ``Gemma3TextModel`` (4.57) op for op on a prompt's path:

* ``embed_tokens``: ``ParallelEmbedding(shard_across_embedding=True)`` (this rank's slice of
  the hidden dimension, then an all-gather), times ``sqrt(hidden_size)`` rounded to the weight
  dtype (``Gemma3TextScaledWordEmbedding``);
* per decoder layer: RMSNorm -> attention -> RMSNorm -> residual -> RMSNorm -> gated tanh-GELU
  MLP -> RMSNorm -> residual (``Gemma3DecoderLayer``). The norms are Gemma's: fp32,
  ``x * rsqrt(mean(x^2) + eps) * (1 + weight)``, cast back. ``q_proj``/``k_proj``/``v_proj``
  are column-parallel without a gather (16 query and 8 key/value heads of 256: at tp4 four and
  two per rank; grouped-query attention keeps query head ``i`` on key/value head ``i // 2``, so a
  contiguous column split keeps every group on one rank), with Gemma's per-head q/k RMSNorm;
  ``o_proj`` and ``down_proj`` are row-parallel, ``gate_proj``/``up_proj`` column-parallel. The
  activation is evaluated in fp32 (the neuron device's bf16 tanh GELU is less accurate than the
  CPU's; see ``difflet/backends/neuron/ltx_2/transformer.py``), the only deviation;
* attention: causal, scaled by ``query_pre_attn_scalar ** -0.5``, through ``difflet.ops.attention``.
  Sliding-window layers are causal too: the window (1024) is not shorter than the sequence the
  encoder is built for, which ``Gemma3TextEncoder`` checks;
* RoPE: the full-attention layers' global table (``rope_theta`` 1e6 with linear scaling by
  ``rope_scaling.factor``, i.e. the inverse frequencies divided by it) and the sliding layers'
  local table (``rope_local_base_freq`` 1e4), each computed once in fp32 for positions
  ``0..max_positions-1`` and cast to the activation dtype, as transformers casts them.

Padding: the prompt arrives RIGHT-padded (its valid tokens first, at positions ``0..n-1``) and is
attended causally without a mask: a valid row only ever attends valid rows before it, so the
valid rows equal transformers' masked encode of the same tokens. LTX-2 pads on the LEFT, where
transformers puts the valid tokens at positions ``L-n..L-1``; RoPE makes the attention a function
of relative positions only, so the valid rows agree up to rounding, and the caller restores the
left-padded layout before the connectors (``_DeviceLtx2TextEncoder``). The padded rows' own outputs
are finite but meaningless; the connectors drop them (they replace padding with registers and
normalise over the valid rows only).

``forward(input_ids)`` returns the 49 hidden states stacked on a new last axis,
``[B, L, hidden, num_layers + 1]``: the embeddings, every layer's output except the last, and the
last layer's output after the final norm (transformers' ``hidden_states`` tuple).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from difflet.common.tp_reduce import config_reduce_dtype
import torch.nn as nn
import torch.nn.functional as F

from difflet.ops import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    attention,
    get_tensor_model_parallel_size,
)

#: Checkpoint prefix of the language model inside ``Gemma3ForConditionalGeneration``.
LANGUAGE_MODEL_PREFIX = "language_model.model."
#: The checkpoint tensors the text encoder never loads.
SKIPPED_PREFIXES = ("vision_tower.", "multi_modal_projector.", "language_model.lm_head.")


@dataclass
class Gemma3TextEncoderConfig:
    """The ``text_config`` of a Gemma-3 checkpoint, the fields the encoder reads."""

    vocab_size: int = 262208
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    rope_scaling_factor: float = 8.0
    query_pre_attn_scalar: float = 256.0
    sliding_window: int = 1024
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    layer_types: list[str] = field(default_factory=list)

    @classmethod
    def from_pretrained(cls, text_encoder_dir) -> "Gemma3TextEncoderConfig":
        raw = json.loads((Path(text_encoder_dir) / "config.json").read_text())
        text = raw.get("text_config", raw)
        scaling = text.get("rope_scaling") or {}
        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type not in ("linear", "default"):
            raise NotImplementedError(f"Gemma-3 rope_scaling type {rope_type!r} is not supported")
        if text.get("attn_logit_softcapping") is not None:
            raise NotImplementedError("Gemma-3 attention logit soft-capping is not supported")
        if text.get("use_bidirectional_attention"):
            raise NotImplementedError("Gemma-3 bidirectional attention is not supported")
        if text.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise NotImplementedError(f"Gemma-3 activation {text['hidden_activation']!r}")
        return cls(
            vocab_size=int(text["vocab_size"]), hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]), rms_norm_eps=float(text["rms_norm_eps"]),
            rope_theta=float(text["rope_theta"]),
            rope_local_base_freq=float(text["rope_local_base_freq"]),
            rope_scaling_factor=float(scaling.get("factor", 1.0)) if rope_type == "linear"
            else 1.0,
            query_pre_attn_scalar=float(text["query_pre_attn_scalar"]),
            sliding_window=int(text["sliding_window"]),
            attention_bias=bool(text.get("attention_bias", False)),
            layer_types=list(text.get("layer_types") or []),
        )


class Gemma3RMSNorm(nn.Module):
    """``Gemma3RMSNorm``: fp32 ``x * rsqrt(mean(x^2) + eps) * (1 + weight)``, cast back."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * (1.0 + self.weight.float())).type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_table(head_dim: int, base: float, factor: float, positions: int):
    """fp32 ``(cos, sin)`` ``[positions, head_dim]`` of transformers' default / linear RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    inv_freq = inv_freq / factor
    pos = torch.arange(positions, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3TextEncoderConfig, tp: int) -> None:
        super().__init__()
        self.head_dim = int(config.head_dim)
        self.heads = int(config.num_attention_heads) // tp
        self.kv_heads = int(config.num_key_value_heads) // tp
        self.groups = self.heads // self.kv_heads
        self.scaling = float(config.query_pre_attn_scalar) ** -0.5
        hidden, bias = int(config.hidden_size), bool(config.attention_bias)
        q_out = int(config.num_attention_heads) * self.head_dim
        kv_out = int(config.num_key_value_heads) * self.head_dim
        self.q_proj = ColumnParallelLinear(hidden, q_out, bias=bias, gather_output=False)
        self.k_proj = ColumnParallelLinear(hidden, kv_out, bias=bias, gather_output=False)
        self.v_proj = ColumnParallelLinear(hidden, kv_out, bias=bias, gather_output=False)
        self.o_proj = RowParallelLinear(q_out, hidden, bias=bias, input_is_parallel=True,
                                        reduce_dtype=config_reduce_dtype(config))
        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = cos[None, None], sin[None, None]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if self.groups > 1:
            k = k[:, :, None].expand(b, self.kv_heads, self.groups, s, self.head_dim).reshape(
                b, self.heads, s, self.head_dim)
            v = v[:, :, None].expand(b, self.kv_heads, self.groups, s, self.head_dim).reshape(
                b, self.heads, s, self.head_dim)
        out = attention(q, k, v, scale=self.scaling, causal=True, tp_q=True, tp_k=True,
                        tp_out=False)
        out = out.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        return self.o_proj(out)


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextEncoderConfig) -> None:
        super().__init__()
        hidden, inter = int(config.hidden_size), int(config.intermediate_size)
        self.gate_proj = ColumnParallelLinear(hidden, inter, bias=False, gather_output=False)
        self.up_proj = ColumnParallelLinear(hidden, inter, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(inter, hidden, bias=False, input_is_parallel=True,
                                           reduce_dtype=config_reduce_dtype(config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        act = F.gelu(gate.float(), approximate="tanh").to(gate.dtype)
        return self.down_proj(act * self.up_proj(x))


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3TextEncoderConfig, tp: int) -> None:
        super().__init__()
        hidden, eps = int(config.hidden_size), float(config.rms_norm_eps)
        self.self_attn = Gemma3Attention(config, tp)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(hidden, eps)
        self.post_attention_layernorm = Gemma3RMSNorm(hidden, eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(hidden, eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(hidden, eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cos, sin))
        x = x + h
        h = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x + h


class Gemma3TextEncoder(nn.Module):
    """Gemma-3's language model at one static sequence length ``max_positions``; see the module
    docstring. ``num_layers`` builds the first N decoder layers (parity runs)."""

    def __init__(self, config: Gemma3TextEncoderConfig, *, max_positions: int,
                 num_layers: int | None = None) -> None:
        super().__init__()
        tp = int(get_tensor_model_parallel_size())
        for what, n in (("query", config.num_attention_heads),
                        ("key/value", config.num_key_value_heads)):
            if int(n) % tp:
                raise ValueError(f"Gemma-3 has {n} {what} heads, which does not divide tp={tp}")
        if int(max_positions) > int(config.sliding_window):
            raise NotImplementedError(
                f"the Gemma-3 encoder runs its sliding-window layers as plain causal attention, "
                f"exact only up to the window ({config.sliding_window} tokens); got "
                f"max_positions={max_positions}"
            )
        self.config = config
        self.max_positions = int(max_positions)
        layers = int(config.num_hidden_layers if num_layers is None else num_layers)
        self.embed_tokens = ParallelEmbedding(int(config.vocab_size), int(config.hidden_size),
                                              shard_across_embedding=True)
        self.embed_scale = math.sqrt(int(config.hidden_size))
        self.layers = nn.ModuleList([Gemma3DecoderLayer(config, tp) for _ in range(layers)])
        self.norm = Gemma3RMSNorm(int(config.hidden_size), eps=config.rms_norm_eps)
        types = list(config.layer_types) or ["full_attention"] * int(config.num_hidden_layers)
        self.layer_is_sliding = [t == "sliding_attention" for t in types[:layers]]
        g_cos, g_sin = rope_table(int(config.head_dim), config.rope_theta,
                                  config.rope_scaling_factor, self.max_positions)
        l_cos, l_sin = rope_table(int(config.head_dim), config.rope_local_base_freq, 1.0,
                                  self.max_positions)
        self.register_buffer("global_cos", g_cos, persistent=False)
        self.register_buffer("global_sin", g_sin, persistent=False)
        self.register_buffer("local_cos", l_cos, persistent=False)
        self.register_buffer("local_sin", l_sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[-1] != self.max_positions:
            raise ValueError(
                f"the Gemma-3 encoder runs at {self.max_positions} tokens, got "
                f"{input_ids.shape[-1]}")
        weight_dtype = self.embed_tokens.weight.dtype
        x = self.embed_tokens(input_ids) * torch.tensor(self.embed_scale, dtype=weight_dtype)
        dtype = x.dtype
        tables = {
            False: (self.global_cos.to(dtype), self.global_sin.to(dtype)),
            True: (self.local_cos.to(dtype), self.local_sin.to(dtype)),
        }
        states = [x]
        for layer, sliding in zip(self.layers, self.layer_is_sliding):
            cos, sin = tables[sliding]
            x = layer(x, cos, sin)
            states.append(x)
        states[-1] = self.norm(states[-1])
        return torch.stack(states, dim=-1)


__all__ = [
    "Gemma3TextEncoder",
    "Gemma3TextEncoderConfig",
    "LANGUAGE_MODEL_PREFIX",
    "SKIPPED_PREFIXES",
    "rope_table",
]
