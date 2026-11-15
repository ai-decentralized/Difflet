# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MiniMax-H3's joint audio/video transformer, as the CUDA backend runs it.

The module tree and parameter names are the released checkpoint's, so ``transformer/`` loads
strictly. The neuron and Trainium backends run their own module
(``difflet/models/minimax_h3/modeling_minimax_h3.py``); neither changes the other's output.

The per-row elementwise work follows the reference's FP32 formulas, with one BF16 rounding
where the reference stores:

- every standalone RMS norm (the token refiner's norms and Q/K norms, the output norm) in
  FP32, as the ``difflet_h3::rms_norm`` custom op;
- in each main block, RMS norm + AdaLN modulation, the gated residuals and the SwiGLU gate
  as fused ``difflet_h3::`` kernels (``fused_ops``), and per-head Q/K RMS norm + rotary
  embedding as one more;
- the output norm's modulation in FP32.

Compiled blocks call the same custom ops, so eager and compiled runs share these kernels;
Inductor would otherwise fuse the norms' reductions in its own order. A request is one packed
sequence without padding rows, so attention takes no mask and SDPA can pick its fastest kernel.
"""

from __future__ import annotations

import inspect
import math

import torch
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn
from torch.nn import functional as F

from difflet.backends.cuda.minimax_h3 import fused_ops  # noqa: F401  (registers difflet_h3::)

#: AdaLN rows per timestep, one per modality tag: 0 video, 1 text, 2 audio.
MODALITIES = 3


@torch.library.custom_op("difflet_h3::rms_norm", mutates_args=())
def _rms_norm_op(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Statistics, scaling and the weight product in FP32; one cast back."""
    value = hidden_states.float()
    normalized = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * normalized).to(hidden_states.dtype)


@_rms_norm_op.register_fake
def _(hidden_states, weight, eps):
    return torch.empty_like(hidden_states)


class RMSNorm(nn.Module):
    """An RMS norm computed in FP32 (``difflet_h3::rms_norm``)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.difflet_h3.rms_norm(hidden_states, self.weight, self.eps)


class _RotaryEmbedding(nn.Module):
    """3-axis rotary tables over the packed ``(t, h, w)`` grid: one FP32 ``inv_freq`` of
    ``rope_freq_dim`` frequencies shared by the axes, the three angle blocks concatenated and
    repeated for the rotate-half convention."""

    def __init__(self, rope_freq_dim: int, rope_theta: float):
        super().__init__()
        exponents = torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)
        self.register_buffer("inv_freq", 1.0 / (rope_theta**exponents), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        freqs = torch.cat(freqs.unbind(dim=1), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class _Attention(nn.Module):
    """Full self-attention over the packed sequence. With rotary tables (main blocks), each
    head's Q/K RMS norm and rotation are one fused kernel."""

    def __init__(self, hidden_size: int, heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads = heads
        inner_dim = heads * head_dim
        self.to_q = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, inner_dim, bias=False)
        self.norm_q = RMSNorm(head_dim, eps)
        self.norm_k = RMSNorm(head_dim, eps)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, hidden_size, bias=False)])

    def forward(self, hidden_states: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        if rotary_emb is None:
            query, key = self.norm_q(query), self.norm_k(key)
        else:
            head_norm_rotary = torch.ops.difflet_h3.head_norm_rotary
            query = head_norm_rotary(query, self.norm_q.weight, self.norm_q.eps, *rotary_emb)
            key = head_norm_rotary(key, self.norm_k.weight, self.norm_k.eps, *rotary_emb)
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
        return self.to_out[0](hidden_states.flatten(2, 3).type_as(query))


class _TokenRefinerBlock(nn.Module):
    """Pre-norm block over the projected text: no AdaLN, no rotary embedding."""

    def __init__(self, hidden_size, heads, head_dim, ffn_dim, norm_eps, qk_norm_eps):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, norm_eps)
        self.attn = _Attention(hidden_size, heads, head_dim, qk_norm_eps)
        self.norm2 = RMSNorm(hidden_size, norm_eps)
        self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, activation_fn="swiglu", bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.ff(self.norm2(hidden_states))


class _TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads,
        head_dim,
        ffn_dim,
        num_layers,
        norm_eps,
        qk_norm_eps,
        final_norm_eps,
    ):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            _TokenRefinerBlock(hidden_size, heads, head_dim, ffn_dim, norm_eps, qk_norm_eps)
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(hidden_size, final_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class _AdaLNModulation(nn.Module):
    """One block's six modulation tables, ``(timesteps * MODALITIES, hidden_size)`` each, in
    ``shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`` order. The activation
    runs at the FP32 timestep embedding's precision; only its result is cast to BF16."""

    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, 6 * hidden_size * MODALITIES, bias=True)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(F.silu(temb).to(self.linear.weight.dtype))
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class _TransformerBlock(nn.Module):
    """Self-attention and SwiGLU feed-forward, each AdaLN-modulated per packed row from the
    ``(timestep, modality)`` tables, with the fused per-row kernels."""

    def __init__(
        self, hidden_size, heads, head_dim, ffn_dim, time_embed_dim, norm_eps, qk_norm_eps
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, norm_eps)
        self.attn = _Attention(hidden_size, heads, head_dim, qk_norm_eps)
        self.norm2 = RMSNorm(hidden_size, norm_eps)
        self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, activation_fn="swiglu", bias=False)
        self.adaln_proj = _AdaLNModulation(time_embed_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        ops = torch.ops.difflet_h3
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        normed = ops.rms_norm_modulate(
            hidden_states, self.norm1.weight, self.norm1.eps, shift_msa, scale_msa, adaln_indices
        )
        attn_output = self.attn(normed, rotary_emb)
        hidden_states = ops.gated_residual(hidden_states, gate_msa, adaln_indices, attn_output)
        normed = ops.rms_norm_modulate(
            hidden_states, self.norm2.weight, self.norm2.eps, shift_mlp, scale_mlp, adaln_indices
        )
        ff_output = ops.swiglu(self.ff.net[0].proj(normed))
        for module in self.ff.net[1:]:
            ff_output = module(ff_output)
        return ops.gated_residual(hidden_states, gate_mlp, adaln_indices, ff_output)


class _AdaLNOut(nn.Module):
    """The output norm, shift/scale-modulated per row from one table row per timestep, with
    the modulation in FP32."""

    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def forward(self, hidden_states, temb, timestep_indices) -> torch.Tensor:
        shift, scale = self.linear(F.silu(temb).to(self.linear.weight.dtype)).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        scale = scale.index_select(0, timestep_indices).float()
        shift = shift.index_select(0, timestep_indices).float()
        return (hidden_states.float() * (1.0 + scale) + shift).to(hidden_states.dtype)


class CudaMiniMaxH3Transformer(nn.Module):
    """One stack of blocks over one packed sequence of text, keyframe, audio and video rows.

    The caller builds the packed layout (``contracts.build_t2va_layout``). ``forward`` returns
    the video and audio velocities for the rows ``video_indices`` and ``audio_indices`` name,
    in that order. The checkpoint is mixed-precision: the patch projections, the timestep MLP
    and the output heads are FP32, everything else BF16.
    """

    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
    ):
        super().__init__()
        heads, head_dim = num_attention_heads, attention_head_dim
        video_patch_dim = in_channels * math.prod(patch_size)
        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)
        self.time_proj = Timesteps(
            num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=freq_dim, time_embed_dim=time_embed_hidden_dim, out_dim=time_embed_dim
        )
        self.rope = _RotaryEmbedding(rope_freq_dim, rope_theta)
        self.token_refiner = _TokenRefiner(
            hidden_size,
            heads,
            head_dim,
            ffn_dim,
            num_refiner_layers,
            norm_eps,
            qk_norm_eps,
            final_norm_eps,
        )
        self.transformer_blocks = nn.ModuleList(
            _TransformerBlock(
                hidden_size, heads, head_dim, ffn_dim, time_embed_dim, norm_eps, qk_norm_eps
            )
            for _ in range(num_layers)
        )
        self.norm_out = _AdaLNOut(hidden_size, time_embed_dim, final_norm_eps)
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, audio_in_channels, bias=True)

    @classmethod
    def from_config(cls, config: dict) -> "CudaMiniMaxH3Transformer":
        """Build from the checkpoint's ``config.json``; keys this module has no use for are
        ignored."""
        names = inspect.signature(cls.__init__).parameters
        return cls(**{key: value for key, value in config.items() if key in names})

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``timestep`` holds the step's distinct timesteps and ``timestep_indices`` each
        row's entry in it; ``token_tags`` are the rows' modalities and ``position_ids`` their
        ``(t, h, w)`` coordinates. Each input is cast to its projection's dtype; the text
        stream sets the packed sequence's."""
        rotary_emb = self.rope(position_ids)
        video = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text = self.token_refiner(text)
        sequence = text.new_zeros((text.shape[0], position_ids.shape[0], text.shape[-1]))
        sequence = sequence.index_copy(1, text_indices, text)
        sequence = sequence.index_copy(1, video_indices, video.to(text.dtype))
        sequence = sequence.index_copy(1, audio_indices, audio.to(text.dtype))

        # One FP32 embedding per distinct timestep, shared by every AdaLN projection.
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
        adaln_indices = timestep_indices * MODALITIES + token_tags
        for block in self.transformer_blocks:
            sequence = block(sequence, temb, adaln_indices, rotary_emb)

        sequence = self.norm_out(sequence, temb, timestep_indices).to(self.proj_out.weight.dtype)
        video = self.proj_out(sequence).index_select(1, video_indices)
        audio = self.audio_proj_out(sequence).index_select(1, audio_indices)
        return video, audio


__all__ = ["CudaMiniMaxH3Transformer", "MODALITIES", "RMSNorm"]
