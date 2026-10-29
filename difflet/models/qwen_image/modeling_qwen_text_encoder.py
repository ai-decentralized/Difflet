"""Qwen2.5-VL's language model, text-only, as Qwen-Image's prompt encoder. Hardware-agnostic.

Qwen-Image conditions its DiT on the last hidden state (after the final RMSNorm) of Qwen2.5-VL's
language model for the templated prompt (diffusers ``QwenImagePipeline._get_qwen_prompt_embeds``:
``hidden_states[-1]``). A prompt is text only, so the vision tower, the image-token merge and the
152k-vocabulary ``lm_head`` never run; this module is just the decoder stack, with the TP layers
of ``difflet.ops`` so the neuron backend can shard the 7.07 B parameters (13.2 GiB in bf16)
over the TP group. Imports only ``torch``, stdlib and ``difflet.ops``.

The module mirrors transformers' ``Qwen2_5_VLTextModel`` (4.57) op for op on the path a prompt
takes:

* ``embed_tokens``: ``ParallelEmbedding(shard_across_embedding=True)`` (lookup of this rank's
  slice of the hidden dimension, then an all-gather), as Wan's umT5 shards its embedding;
* per decoder layer: RMSNorm -> attention -> residual -> RMSNorm -> gated SiLU MLP -> residual.
  ``q_proj``/``k_proj``/``v_proj`` (with bias) are column-parallel without a gather, so each
  rank holds ``heads / tp`` query heads and ``kv_heads / tp`` key/value heads (28 / 4 and
  4 / 4 at tp4): grouped-query attention keeps query head ``i`` on key/value head
  ``i // (heads / kv_heads)``, so a contiguous column split keeps every group on one rank.
  ``o_proj`` and ``down_proj`` are row-parallel (one all-reduce each), ``gate_proj``/``up_proj``
  column-parallel;
* RMSNorm in fp32 with the weight applied in the input dtype (``Qwen2RMSNorm``).

Rotary embedding: Qwen2.5-VL's multimodal RoPE (``apply_multimodal_rotary_pos_emb``) splits the
head dimension into ``mrope_section`` chunks (16/24/24 frequency pairs) and takes each chunk's
angle from one of three position axes (temporal, height, width). For text, transformers gives the
three axes the same positions (``get_rope_index`` without image grids: ``cumsum(mask) - 1``, i.e.
``0..n-1`` on a right-padded prompt), so every chunk takes the same angle and the result is the
ordinary 1-D RoPE with ``rope_theta``: ``x * cos + rotate_half(x) * sin`` over ``cat(freqs,
freqs)``. That is what this module computes, once, for positions ``0..max_positions-1``, in fp32
(transformers computes the angles in fp32 with autocast off and casts cos/sin to the activation
dtype, as here). The equivalence is tested against transformers itself.

Attention mask: none. The prompt is padded on the right to a static length and attention is
causal, so a valid token only ever attends valid tokens before it; padding positions influence
nothing that is kept (Qwen-Image keeps the valid rows only, ``qwen_prompt_embeds``). Causal
attention alone therefore reproduces the masked encode on every valid row, and it is what the
neuron flash kernel takes (``attention(causal=True)``: the kernel's ``is_causal``; it takes no
mask, and a mask would send the call through SDPA). The padded rows' own outputs are finite
but meaningless (they attend the padding before them) and are dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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


def _safe_tp_size() -> int:
    """TP degree, or 1 without a parallel mesh (a CPU reference build)."""
    try:
        return int(get_tensor_model_parallel_size())
    except Exception:
        return 1


@dataclass(frozen=True)
class QwenTextEncoderConfig:
    """The language-model hyperparameters a prompt encode needs (``text_config`` of the
    checkpoint's ``config.json``). Defaults are Qwen-Image's Qwen2.5-VL-7B."""

    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate(self, tp_degree: int = 1) -> None:
        heads, kv, tp = self.num_attention_heads, self.num_key_value_heads, int(tp_degree)
        if self.head_dim * heads != self.hidden_size:
            raise ValueError(
                f"Qwen2.5-VL hidden_size={self.hidden_size} is not num_attention_heads={heads} x "
                "an integer head_dim"
            )
        if heads % kv:
            raise ValueError(f"Qwen2.5-VL has {heads} query heads, not a multiple of {kv} kv heads")
        if self.hidden_act != "silu":
            raise NotImplementedError(f"Qwen2.5-VL hidden_act={self.hidden_act!r}; only silu")
        # The column-parallel q/k/v split heads over tp, the embedding splits the hidden size
        # and the MLP the intermediate size; a ragged split fails deep in the load.
        for name, value in (("num_attention_heads", heads), ("num_key_value_heads", kv),
                            ("hidden_size", self.hidden_size),
                            ("intermediate_size", self.intermediate_size)):
            if value % tp:
                raise ValueError(
                    f"Qwen2.5-VL {name}={value} does not divide tp={tp} (the device text encoder "
                    "shards it over the TP group)"
                )

    @classmethod
    def from_dict(cls, raw: dict) -> "QwenTextEncoderConfig":
        """From a Qwen2.5-VL ``config.json`` (its ``text_config``, or the top level of an older
        file). Rejects what this module does not implement: a RoPE scaling other than the
        default (mrope reduces to it for text) and sliding-window attention."""
        text = dict(raw.get("text_config") or raw)
        scaling = text.get("rope_scaling") or {}
        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type not in ("default", "mrope"):
            raise NotImplementedError(
                f"Qwen2.5-VL rope_scaling type {rope_type!r}; the device text encoder implements "
                "the default rotary embedding only"
            )
        if text.get("use_sliding_window"):
            raise NotImplementedError("Qwen2.5-VL with use_sliding_window=True is not supported")
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in text.items() if k in names})

    @classmethod
    def from_pretrained(cls, text_encoder_dir) -> "QwenTextEncoderConfig":
        raw = json.loads((Path(text_encoder_dir) / "config.json").read_text())
        return cls.from_dict(raw)


def rotary_tables(head_dim: int, positions: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(cos, sin)`` of shape ``[positions, head_dim]`` for 1-D positions ``0..n-1``:
    transformers' ``Qwen2_5_VLRotaryEmbedding`` (default rope, attention scaling 1) on a text
    prompt, whose three mrope axes carry the same positions."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32)
                  / head_dim)
    )
    position = torch.arange(positions, dtype=torch.float32)
    freqs = (inv_freq[:, None] @ position[None, :]).transpose(0, 1)  # as transformers, a matmul
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """``[B, kv, S, d]`` -> ``[B, kv * groups, S, d]``, each kv head repeated for its group (the
    flash kernel wants as many key/value heads as query heads)."""
    if groups == 1:
        return x
    b, kv, s, d = x.shape
    return x[:, :, None].expand(b, kv, groups, s, d).reshape(b, kv * groups, s, d)


class QwenTextRMSNorm(nn.Module):
    """``Qwen2RMSNorm``: normalise in fp32, apply the weight in the input dtype."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        hidden = x.to(torch.float32)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.eps)
        return self.weight * hidden.to(dtype)


class QwenTextAttention(nn.Module):
    """Causal grouped-query attention over this rank's heads (see the module docstring)."""

    def __init__(self, config: QwenTextEncoderConfig, tp: int):
        super().__init__()
        d, hd = config.hidden_size, config.head_dim
        self.head_dim = hd
        self.heads = config.num_attention_heads // tp
        self.kv_heads = config.num_key_value_heads // tp
        self.groups = self.heads // self.kv_heads
        self.scale = hd**-0.5
        self.q_proj = ColumnParallelLinear(d, config.num_attention_heads * hd, bias=True,
                                           gather_output=False)
        self.k_proj = ColumnParallelLinear(d, config.num_key_value_heads * hd, bias=True,
                                           gather_output=False)
        self.v_proj = ColumnParallelLinear(d, config.num_key_value_heads * hd, bias=True,
                                           gather_output=False)
        self.o_proj = RowParallelLinear(config.num_attention_heads * hd, d, bias=False,
                                        input_is_parallel=True,
                                        reduce_dtype=config_reduce_dtype(config))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        k, v = _repeat_kv(k, self.groups), _repeat_kv(v, self.groups)
        out = attention(q, k, v, scale=self.scale, causal=True, tp_q=True, tp_k=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, self.heads * self.head_dim))


class QwenTextMLP(nn.Module):
    def __init__(self, config: QwenTextEncoderConfig):
        super().__init__()
        d, ff = config.hidden_size, config.intermediate_size
        self.gate_proj = ColumnParallelLinear(d, ff, bias=False, gather_output=False)
        self.up_proj = ColumnParallelLinear(d, ff, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(ff, d, bias=False, input_is_parallel=True,
                                           reduce_dtype=config_reduce_dtype(config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenTextDecoderLayer(nn.Module):
    """One ``Qwen2_5_VLDecoderLayer``; every layer has the same graph (the compile unit)."""

    def __init__(self, config: QwenTextEncoderConfig, tp: int):
        super().__init__()
        self.self_attn = QwenTextAttention(config, tp)
        self.mlp = QwenTextMLP(config)
        self.input_layernorm = QwenTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = QwenTextRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class QwenTextEncoderModel(nn.Module):
    """``input_ids [B, S]`` (S <= ``max_positions``, right-padded) -> the last hidden state
    ``[B, S, hidden]`` after the final norm. Parameter names are the checkpoint's without its
    ``model.`` prefix (``embed_tokens.weight``, ``layers.{i}.self_attn.q_proj.weight``, ...)."""

    def __init__(self, config: QwenTextEncoderConfig, *, max_positions: int):
        super().__init__()
        tp = _safe_tp_size()
        config.validate(tp)
        self.config = config
        self.embed_tokens = ParallelEmbedding(config.vocab_size, config.hidden_size,
                                              shard_across_embedding=True)
        self.layers = nn.ModuleList(
            [QwenTextDecoderLayer(config, tp) for _ in range(config.num_hidden_layers)]
        )
        self.norm = QwenTextRMSNorm(config.hidden_size, config.rms_norm_eps)
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
        return self.norm(x)


__all__ = [
    "QwenTextEncoderConfig",
    "QwenTextEncoderModel",
    "rotary_tables",
]
