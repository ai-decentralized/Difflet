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

"""MiniMax-H3's audio VAE decoder, as the CUDA backend runs it: a BigVGAN vocoder with
anti-aliased SnakeBeta activations, FP32 weights in the checkpoint's weight-norm form.

The module tree and parameter names are the released ``audio_vae/`` checkpoint's. The neuron
and Trainium backends run their own module (``difflet/models/minimax_h3/modeling_vae.py``);
neither changes the other's output. Each anti-aliased activation upsamples 2x with a
transposed convolution, applies SnakeBeta and low-pass filters back down.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from difflet.backends.cuda.minimax_h3.vae import load_fp32


def _kaiser_sinc_filter(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """A Kaiser-windowed sinc low-pass filter ``(1, 1, kernel_size)``, normalized to sum 1."""
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


class _SnakeBeta(nn.Module):
    """``x + sin^2(alpha * x) / beta`` per channel, with log-scale ``alpha`` and ``beta``."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)


class _LowPassFilter(nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", _kaiser_sinc_filter(cutoff, half_width, kernel_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )


class _UpSample(nn.Module):
    """Zero-stuffing by ``ratio`` and the low-pass filter, as one transposed convolution."""

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer("filter", _kaiser_sinc_filter(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states, self.filter.expand(channels, -1, -1), stride=self.ratio, groups=channels
        )
        return hidden_states[..., self.pad_left : -self.pad_right]


class _DownSample(nn.Module):
    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = _LowPassFilter(0.5 / ratio, 0.6 / ratio, ratio, kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lowpass(hidden_states)


class _AntiAliasedActivation(nn.Module):
    def __init__(self, channels: int, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = _SnakeBeta(channels)
        self.upsample = _UpSample(ratio, kernel_size)
        self.downsample = _DownSample(ratio, kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(hidden_states)))


class _AMPBlock(nn.Module):
    """Residual pairs of (dilated, plain) convolutions, each after an anti-aliased activation."""

    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=value,
                    padding=(kernel_size * value - value) // 2,
                )
            )
            for value in dilation
        )
        self.convs2 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2))
            for _ in dilation
        )
        self.activations = nn.ModuleList(
            _AntiAliasedActivation(channels) for _ in range(2 * len(dilation))
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(
            self.convs1, self.convs2, self.activations[::2], self.activations[1::2]
        ):
            hidden_states = hidden_states + conv2(act2(conv1(act1(hidden_states))))
        return hidden_states


class _BigVGAN(nn.Module):
    def __init__(self, c):
        super().__init__()
        initial = c.decoder_dim
        self.num_kernels = len(c.resblock_kernel_sizes)
        self.num_upsamples = len(c.decoder_rates)
        self.conv_pre = weight_norm(nn.Conv1d(c.latent_dim, initial, 7, 1, padding=3))
        self.ups = nn.ModuleList(
            nn.ModuleList(
                [
                    weight_norm(
                        nn.ConvTranspose1d(
                            initial // 2**index,
                            initial // 2 ** (index + 1),
                            kernel,
                            rate,
                            padding=(kernel - rate) // 2,
                        )
                    )
                ]
            )
            for index, (rate, kernel) in enumerate(zip(c.decoder_rates, c.decoder_kernel_sizes))
        )
        self.resblocks = nn.ModuleList(
            _AMPBlock(initial // 2 ** (index + 1), kernel, tuple(dilation))
            for index in range(self.num_upsamples)
            for kernel, dilation in zip(c.resblock_kernel_sizes, c.resblock_dilation_sizes)
        )
        channels = initial // 2**self.num_upsamples
        self.activation_post = _AntiAliasedActivation(channels)
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False))

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


class AudioDecoder(nn.Module):
    """Stereo-as-batch latents ``(2, C, T)`` (on any device) to mono waveforms ``(2, 1,
    samples)`` on the decoder's device."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = SimpleNamespace(**config)
        self.dec_in_proj = nn.Conv1d(self.config.latent_channels, self.config.latent_dim, 1)
        # The checkpoint's weight_g/weight_v: the weight-norm hooks recompute each weight
        # from the loaded FP32 parameters.
        self.decoder = _BigVGAN(self.config)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.dec_in_proj.weight.device)
        return self.decoder(self.dec_in_proj(latents))


@torch.no_grad()
def decode_audio(decoder: AudioDecoder, normalized_latents: torch.Tensor) -> torch.Tensor:
    """``(1, 2, samples)`` in [-1, 1] on the decoder's device, from normalized stereo-as-batch
    ``(2, C, T)`` latents."""
    c = decoder.config
    device = normalized_latents.device
    mean = torch.tensor(c.latents_mean, dtype=torch.float32, device=device).view(1, -1, 1)
    std = torch.tensor(c.latents_std, dtype=torch.float32, device=device).view(1, -1, 1)
    waveform = decoder(normalized_latents.float() * std + mean)
    return waveform.float().permute(1, 0, 2).contiguous()


def load_audio_decoder(model_path, *, device) -> AudioDecoder:
    return load_fp32(AudioDecoder, model_path, device=device)


__all__ = ["AudioDecoder", "decode_audio", "load_audio_decoder"]
