# Copyright 2025-2026 The MiniMax authors and The HuggingFace Team. All rights reserved.
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

"""Decoder-only MiniMax-H3 visual and audio VAE modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from torch.nn.utils import weight_norm


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"dim {dim} must be divisible by {2 * num_axes}")
        inv_freq = 1.0 / theta ** torch.arange(
            0,
            1,
            2 * num_axes / dim,
            dtype=torch.float32,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


def apply_h3_video_rotary(
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    if rotary_emb is None:
        return hidden_states
    cos, sin = rotary_emb
    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat([-second, first], dim=-1)
    return torch.cat([rotary * cos + rotated * sin, passthrough], dim=-1)


class MiniMaxH3VideoAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "MiniMaxH3VideoAttention",
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        query = attn.norm_q(query.float()).to(query.dtype)
        key = attn.norm_k(key.float()).to(key.dtype)
        query = apply_h3_video_rotary(query, rotary_emb)
        key = apply_h3_video_rotary(key, rotary_emb)

        if attention_mask is None:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=None,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attention_mask[:, None, None, :],
            ).transpose(1, 2)
        return attn.to_out[0](hidden_states.flatten(2, 3))


class MiniMaxH3VideoAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.use_bias = bias
        inner_dim = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=bias), nn.Dropout(0.0)])
        self.processor = MiniMaxH3VideoAttnProcessor()

    def set_processor(self, processor) -> None:
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb, attention_mask)


class MiniMaxH3VideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3VideoAttention(dim, heads, dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = FeedForward(dim, mult=ffn_mult, activation_fn="swiglu", bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = (
            hidden_states
            + self.attn(
                norm_hidden_states,
                rotary_emb,
                attention_mask,
            )
            * self.scale1
        )
        norm_hidden_states = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        return hidden_states + self.ff(norm_hidden_states) * self.scale2


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    """Official H3 ViT decoder with optional tail alignment for attention_cte."""

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        norm_eps: float = 1e-5,
        sequence_alignment: int = 1,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.sequence_alignment = sequence_alignment
        self.rope = MiniMaxH3VideoRotaryPosEmbed(
            int(attention_head_dim * rope_dim_ratio),
            theta=rope_theta,
        )
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(
            dim,
            out_channels * patch_size_t * patch_size * patch_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size,
            num_frames * height * width,
            num_channels,
        )
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)
        valid_tokens = hidden_states.shape[1]

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size)
            - 1.0
            for size in (num_frames, height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)

        aligned_tokens = (
            (valid_tokens + self.sequence_alignment - 1) // self.sequence_alignment
        ) * self.sequence_alignment
        padding = aligned_tokens - valid_tokens
        attention_mask = None
        if padding:
            hidden_states = torch.cat(
                [
                    hidden_states,
                    hidden_states.new_zeros(batch_size, padding, hidden_states.shape[-1]),
                ],
                dim=1,
            )
            position_ids = torch.cat(
                [position_ids, position_ids.new_zeros(batch_size, padding, 3)],
                dim=1,
            )
            attention_mask = torch.zeros(
                batch_size,
                aligned_tokens,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            attention_mask[:, :valid_tokens] = True
        rotary_emb = self.rope(position_ids)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb, attention_mask)

        hidden_states = self.proj_out(self.norm_out(hidden_states))[:, :num_patches, :]
        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class MiniMaxH3AudioSnakeBeta(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)


class MiniMaxH3AudioLowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(
            hidden_states,
            (self.pad_left, self.pad_right),
            mode="replicate",
        )
        return F.conv1d(
            hidden_states,
            self.filter.expand(num_channels, -1, -1),
            stride=self.stride,
            groups=num_channels,
        )


class MiniMaxH3AudioUpSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        filter_ = self.filter.expand(num_channels, -1, -1)
        if self.ratio == 2:
            # Exact polyphase form of the stride-two transposed convolution.
            # neuronx-cc otherwise materializes an expanded depthwise-conv
            # scratch tensor that exceeds SB for useful audio lengths.
            phase_padding = filter_.shape[-1] // 2 - 1
            even = F.conv1d(
                hidden_states,
                filter_[..., 0::2].flip(-1),
                padding=phase_padding,
                groups=num_channels,
            )
            odd = F.conv1d(
                hidden_states,
                filter_[..., 1::2].flip(-1),
                padding=phase_padding,
                groups=num_channels,
            )
            hidden_states = self.ratio * torch.stack((even, odd), dim=-1).flatten(-2)
        else:
            hidden_states = self.ratio * F.conv_transpose1d(
                hidden_states,
                filter_,
                stride=self.stride,
                groups=num_channels,
            )
        return hidden_states[..., self.pad_left : -self.pad_right]


class MiniMaxH3AudioDownSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = MiniMaxH3AudioLowPassFilter1d(
            0.5 / ratio,
            0.6 / ratio,
            ratio,
            kernel_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lowpass(hidden_states)


class MiniMaxH3AudioActivation1d(nn.Module):
    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = MiniMaxH3AudioUpSample1d(ratio, kernel_size)
        self.downsample = MiniMaxH3AudioDownSample1d(ratio, kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(hidden_states)))


class MiniMaxH3AudioAMPBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: tuple[int, ...],
        *,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        wrap = weight_norm if use_weight_norm else (lambda module: module)
        self.convs1 = nn.ModuleList(
            [
                wrap(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=value,
                        padding=(kernel_size * value - value) // 2,
                    )
                )
                for value in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                wrap(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=1,
                        padding=(kernel_size - 1) // 2,
                    )
                )
                for _ in dilation
            ]
        )
        self.activations = nn.ModuleList(
            [
                MiniMaxH3AudioActivation1d(MiniMaxH3AudioSnakeBeta(channels))
                for _ in range(2 * len(dilation))
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(
            self.convs1,
            self.convs2,
            self.activations[::2],
            self.activations[1::2],
        ):
            hidden_states = hidden_states + conv2(act2(conv1(act1(hidden_states))))
        return hidden_states


class MiniMaxH3AudioBigVGANDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        wrap = weight_norm if use_weight_norm else (lambda module: module)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = wrap(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        wrap(
                            nn.ConvTranspose1d(
                                upsample_initial_channel // (2**index),
                                upsample_initial_channel // (2 ** (index + 1)),
                                kernel,
                                rate,
                                padding=(kernel - rate) // 2,
                            )
                        )
                    ]
                )
            )
        self.resblocks = nn.ModuleList()
        for index in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(
                    MiniMaxH3AudioAMPBlock(
                        channels,
                        kernel,
                        tuple(dilation),
                        use_weight_norm=use_weight_norm,
                    )
                )
        self.activation_post = MiniMaxH3AudioActivation1d(MiniMaxH3AudioSnakeBeta(channels))
        self.conv_post = wrap(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_pre(hidden_states)
        for index in range(self.num_upsamples):
            hidden_states = self.ups[index][0](hidden_states)
            residual = None
            for kernel_index in range(self.num_kernels):
                block = self.resblocks[index * self.num_kernels + kernel_index](hidden_states)
                residual = block if residual is None else residual + block
            hidden_states = residual / self.num_kernels
        hidden_states = self.conv_post(self.activation_post(hidden_states))
        return torch.clamp(hidden_states, min=-1.0, max=1.0)
