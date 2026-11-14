"""Qwen3-VL's language model, text-only, as MiniMax-H3's conditioner. Hardware-agnostic.

H3 conditions its DiT on ``hidden_states[50]`` of Qwen3-VL-32B's language model: the output of
decoder layer 49, before any later layer and without the final norm. This module is therefore
the embedding plus the first ``num_layers`` (50) decoder layers, with the TP layers of
``difflet.ops`` so the neuron backend can shard the ~25 B parameters (about 50 GB in bf16, 12.6 GB
per rank at tp4). Layers 50-63, the final norm, ``lm_head`` and the vision tower are never built.

It mirrors transformers' ``Qwen3VLTextModel`` (4.57) on the path a text prompt takes, and reuses
the Qwen2.5-VL encoder's pieces (``difflet/models/qwen_image/modeling_qwen_text_encoder.py``):

* ``embed_tokens``: ``ParallelEmbedding(shard_across_embedding=True)``;
* per layer: RMSNorm -> attention -> residual -> RMSNorm -> gated SiLU MLP -> residual;
* attention (``Qwen3VLTextAttention``): q/k/v without bias, column-parallel (64 / tp query heads,
  8 / tp key/value heads, grouped as in Qwen2.5), a per-head RMSNorm on q and on k
  (``q_norm``/``k_norm``, over ``head_dim`` = 128, which is not ``hidden_size / heads``), RoPE,
  causal attention, row-parallel ``o_proj``;
* RoPE: Qwen3-VL's interleaved multimodal RoPE gives a text prompt's three axes the same
  positions, so every frequency takes the same angle and the result is the ordinary 1-D RoPE with
  ``rope_theta`` (5e6) over ``head_dim``; computed once in fp32 for ``0..max_positions-1``.

No attention mask: the prompt is right-padded and attention is causal, so the valid rows never
see the padding (the padded rows' outputs are dropped by the caller), exactly as the Trainium
stage's NxDI encode and the host encoder treat it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Qwen3VLTextEncoderConfig:
    """``text_config`` of the checkpoint's ``config.json``; ``num_layers`` is how many decoder
    layers to build (H3: 50, for ``hidden_states[50]``). Defaults are Qwen3-VL-32B's."""

    vocab_size: int = 151936
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    hidden_act: str = "silu"
    attention_bias: bool = False
    num_layers: int = 50

    def validate(self, tp_degree: int = 1) -> None:
        heads, kv, tp = self.num_attention_heads, self.num_key_value_heads, int(tp_degree)
        if heads % kv:
            raise ValueError(f"Qwen3-VL has {heads} query heads, not a multiple of {kv} kv heads")
        if self.hidden_act != "silu" or self.attention_bias:
            raise NotImplementedError("Qwen3-VL text encoder: silu MLP and bias-free q/k/v only")
        if not 1 <= int(self.num_layers) <= int(self.num_hidden_layers):
            raise ValueError(f"num_layers={self.num_layers} outside 1..{self.num_hidden_layers}")
        for name, value in (("num_attention_heads", heads), ("num_key_value_heads", kv),
                            ("hidden_size", self.hidden_size),
                            ("intermediate_size", self.intermediate_size)):
            if value % tp:
                raise ValueError(f"Qwen3-VL {name}={value} does not divide tp={tp}")

    @classmethod
    def from_pretrained(cls, text_encoder_dir, **overrides) -> "Qwen3VLTextEncoderConfig":
        raw = json.loads((Path(text_encoder_dir) / "config.json").read_text())
        text = dict(raw.get("text_config") or raw)
        scaling = text.get("rope_scaling") or {}
        if scaling.get("rope_type", scaling.get("type", "default")) != "default":
            raise NotImplementedError(f"Qwen3-VL rope_scaling {scaling!r}")
        names = set(cls.__dataclass_fields__)
        values = {k: v for k, v in text.items() if k in names}
        values.update(overrides)
        return cls(**values)


class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config: Qwen3VLTextEncoderConfig, tp: int):
        super().__init__()
        from difflet.models.qwen_image.modeling_qwen_text_encoder import QwenTextRMSNorm
        from difflet.ops import ColumnParallelLinear, RowParallelLinear, attention

        d, hd = config.hidden_size, config.head_dim
        self.head_dim = hd
        self.heads = config.num_attention_heads // tp
        self.kv_heads = config.num_key_value_heads // tp
        self.groups = self.heads // self.kv_heads
        self.scale = hd**-0.5
        self._attention = attention
        self.q_proj = ColumnParallelLinear(d, config.num_attention_heads * hd, bias=False,
                                           gather_output=False)
        self.k_proj = ColumnParallelLinear(d, config.num_key_value_heads * hd, bias=False,
                                           gather_output=False)
        self.v_proj = ColumnParallelLinear(d, config.num_key_value_heads * hd, bias=False,
                                           gather_output=False)
        self.o_proj = RowParallelLinear(config.num_attention_heads * hd, d, bias=False,
                                        input_is_parallel=True, reduce_dtype=torch.bfloat16)
        self.q_norm = QwenTextRMSNorm(hd, config.rms_norm_eps)
        self.k_norm = QwenTextRMSNorm(hd, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        from difflet.models.qwen_image.modeling_qwen_text_encoder import _repeat_kv, _rotate_half

        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, s, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        k, v = _repeat_kv(k, self.groups), _repeat_kv(v, self.groups)
        out = self._attention(q, k, v, scale=self.scale, causal=True, tp_q=True, tp_k=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, self.heads * self.head_dim))


class Qwen3VLTextDecoderLayer(nn.Module):
    """One ``Qwen3VLTextDecoderLayer``; every layer has the same graph (the compile unit)."""

    def __init__(self, config: Qwen3VLTextEncoderConfig, tp: int):
        super().__init__()
        from difflet.models.qwen_image.modeling_qwen_text_encoder import (
            QwenTextMLP,
            QwenTextRMSNorm,
        )

        self.self_attn = Qwen3VLTextAttention(config, tp)
        self.mlp = QwenTextMLP(config)
        self.input_layernorm = QwenTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = QwenTextRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3VLTextEncoderModel(nn.Module):
    """``input_ids [B, S]`` (right-padded, S <= ``max_positions``) -> ``hidden_states[num_layers]``
    ``[B, S, hidden]`` (no final norm). Parameter names are the checkpoint's without its
    ``model.language_model.`` prefix."""

    def __init__(self, config: Qwen3VLTextEncoderConfig, *, max_positions: int):
        super().__init__()
        from difflet.models.qwen_image.modeling_qwen_text_encoder import _safe_tp_size, rotary_tables
        from difflet.ops import ParallelEmbedding

        tp = _safe_tp_size()
        config.validate(tp)
        self.config = config
        self.embed_tokens = ParallelEmbedding(config.vocab_size, config.hidden_size,
                                              shard_across_embedding=True)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, tp) for _ in range(int(config.num_layers))]
        )
        cos, sin = rotary_tables(config.head_dim, int(max_positions), float(config.rope_theta))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        seq = int(input_ids.shape[1])
        cos = self.rope_cos[:seq].to(x.dtype)[None, None]
        sin = self.rope_sin[:seq].to(x.dtype)[None, None]
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


__all__ = ["Qwen3VLTextEncoderConfig", "Qwen3VLTextEncoderModel"]
