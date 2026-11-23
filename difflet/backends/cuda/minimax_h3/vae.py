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

"""MiniMax-H3's visual VAE, as the CUDA backend runs it, with FP32 weights.

The module trees and parameter names are the released ``vae/`` checkpoint's. The neuron and
Trainium backends run their own modules (``difflet/models/minimax_h3/``); neither changes the
other's output.

- ``VideoEncoder``: the keyframe's posterior moments, ``quant_conv(encoder(x))``, over the
  reference's tiles (256 pixels, overlaps of at least 64, linear blends).
- ``VideoDecoder``: ``decoder(post_quant_conv(z))``, the ViT blocks under FP16 autocast as in
  the reference, with the reference's arithmetic around them: the token grid and rotary tables
  in the activation dtype, the input/output projections in FP32 outside autocast.
- ``decode_video``: the reference's temporal chunking and spatial tiling, stitched on the
  latents' device.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.models.attention import FeedForward
from torch import nn
from torch.nn import functional as F

from difflet.backends._torch_megatron.checkpoint import load_checkpoint_into
from difflet.backends._torch_megatron.weights import materialize_meta_

# The reference's ImageNet pixel normalization, undone after decoding.
_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)
_TILE = 256
_TILE_MIN_OVERLAP = 64


# ---------------------------------------------------------------------------------- tiles


def _split_tiles(length: int, *, alignment: int) -> tuple[list[int], list[int], list[int]]:
    """The reference's tile starts, lengths and overlaps along one side."""
    if _TILE >= length:
        return [0], [length], []
    num_tiles = math.ceil(length / _TILE)
    while _TILE * num_tiles - _TILE_MIN_OVERLAP * (num_tiles - 1) < length:
        num_tiles += 1
    overlaps = [_TILE_MIN_OVERLAP] * (num_tiles - 1)
    remaining = _TILE * num_tiles - sum(overlaps) - length
    for index in range(remaining // alignment):
        overlaps[index % (num_tiles - 1)] += alignment
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + _TILE - overlap)
    return starts, [_TILE] * num_tiles, overlaps


def _blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
    """``b`` with its first ``extent`` positions along ``dim`` blended linearly from ``a``'s
    last ones."""
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    weight_a = (1 - positions / extent).view(shape)
    weight_b = (positions / extent).view(shape)
    slice_a = [slice(None)] * a.ndim
    slice_a[dim] = slice(-extent, None)
    slice_b = [slice(None)] * b.ndim
    slice_b[dim] = slice(0, extent)
    blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
    if extent == b.shape[dim]:
        return blended
    slice_rest = [slice(None)] * b.ndim
    slice_rest[dim] = slice(extent, None)
    return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)


def _stitch_tiles(tiles, height_overlaps, width_overlaps) -> torch.Tensor:
    """One tensor from a grid of tiles, each overlap blended into the later tile."""
    result_rows = []
    for row_index, row in enumerate(tiles):
        result_row = []
        for column_index, tile in enumerate(row):
            if row_index > 0:
                tile = _blend(
                    tiles[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2
                )
            if column_index > 0:
                tile = _blend(row[column_index - 1], tile, width_overlaps[column_index - 1], -1)
            if row_index < len(tiles) - 1:
                tile = tile[..., : -height_overlaps[row_index], :]
            if column_index < len(row) - 1:
                tile = tile[..., :, : -width_overlaps[column_index]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


# -------------------------------------------------------------------------------- encoder


class _CausalConv3d(nn.Conv3d):
    """Symmetric spatial padding in ``spatial_padding_mode`` and causal temporal padding
    (``temporal_padding`` zero frames in front)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        spatial_padding=0,
        temporal_padding=0,
        spatial_padding_mode="reflect",
    ):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            pad = self.spatial_padding
            hidden_states = F.pad(
                hidden_states, (pad, pad, pad, pad, 0, 0), mode=self.spatial_padding_mode
            )
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0))
        return F.conv3d(
            hidden_states, self.weight, self.bias, stride=self.stride, dilation=self.dilation
        )


class _FrameGroupNorm(nn.GroupNorm):
    """Group norm with each frame normalized on its own."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


class _ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups, norm_eps, padding_mode):
        super().__init__()
        self.norm1 = _FrameGroupNorm(norm_num_groups, in_channels, eps=norm_eps)
        self.conv1 = _CausalConv3d(
            in_channels,
            out_channels,
            3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=padding_mode,
        )
        self.norm2 = _FrameGroupNorm(norm_num_groups, out_channels, eps=norm_eps)
        self.conv2 = _CausalConv3d(
            out_channels,
            out_channels,
            3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=padding_mode,
        )
        self.conv_shortcut = (
            _CausalConv3d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(F.silu(self.norm2(hidden_states)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class _Downsample(nn.Module):
    """A strided 3x3x3 convolution; a spatial stride of 2 first pads one row and column at the
    bottom/right, so each side becomes ``ceil(size / 2)``."""

    def __init__(self, channels, temporal_stride, spatial_stride, padding_mode):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = padding_mode
        self.conv = _CausalConv3d(
            channels,
            channels,
            3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            temporal_padding=2,
            spatial_padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


class _DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        temporal_factor,
        spatial_factor,
        norm_num_groups,
        norm_eps,
        padding_mode,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            _ResnetBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                norm_num_groups,
                norm_eps,
                padding_mode,
            )
            for i in range(num_layers)
        )
        self.downsamplers = None
        if temporal_factor * spatial_factor > 1:
            self.downsamplers = nn.ModuleList(
                [_Downsample(out_channels, temporal_factor, spatial_factor, padding_mode)]
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        for downsampler in self.downsamplers or ():
            hidden_states = downsampler(hidden_states)
        return hidden_states


class _Encoder(nn.Module):
    """The causal 3D CNN encoder; each level's factors multiply out to the compression."""

    def __init__(self, c: SimpleNamespace):
        super().__init__()
        channels = tuple(c.block_out_channels)
        mode = c.spatial_padding_mode
        self.conv_in = _CausalConv3d(
            c.in_channels,
            channels[0],
            3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=mode,
        )
        inputs = (channels[0],) + channels[:-1]
        self.down_blocks = nn.ModuleList(
            _DownBlock(
                inputs[i],
                channels[i],
                c.layers_per_block,
                c.temporal_downsample_factors[i],
                c.spatial_downsample_factors[i],
                c.norm_num_groups,
                c.norm_eps,
                mode,
            )
            for i in range(len(channels))
        )
        self.norm_out = _FrameGroupNorm(c.norm_num_groups, channels[-1], eps=c.norm_eps)
        self.conv_out = _CausalConv3d(
            channels[-1],
            2 * c.latent_channels,
            3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        return self.conv_out(F.silu(self.norm_out(hidden_states)))


def _namespace(config: dict) -> SimpleNamespace:
    """The VAE config, with the compression ratios its per-level factors multiply out to."""
    config = dict(config)
    for kind in ("spatial", "temporal"):
        factors = config.get(f"{kind}_downsample_factors")
        if factors is not None:
            config[f"{kind}_compression_ratio"] = math.prod(factors)
    return SimpleNamespace(**config)


class VideoEncoder(nn.Module):
    """``quant_conv(encoder(x))`` of one keyframe, tiled as the reference encodes."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = _namespace(config)
        latent = 2 * self.config.latent_channels
        self.encoder = _Encoder(self.config)
        self.quant_conv = nn.Conv3d(latent, latent, kernel_size=1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(pixels))

    def moments(self, pixels: torch.Tensor) -> torch.Tensor:
        """The posterior moments ``(1, 2 * latent_channels, 1, h, w)`` of one normalized FP32
        frame ``(1, 3, 1, H, W)`` on the encoder's device."""
        if pixels.shape[2] != 1:
            raise ValueError("the MiniMax-H3 keyframe encoder takes one frame")
        ratio = self.config.spatial_compression_ratio
        ys, y_lengths, y_overlaps = _split_tiles(pixels.shape[-2], alignment=ratio)
        xs, x_lengths, x_overlaps = _split_tiles(pixels.shape[-1], alignment=ratio)
        rows = [
            [self(pixels[..., y : y + h, x : x + w]) for x, w in zip(xs, x_lengths)]
            for y, h in zip(ys, y_lengths)
        ]
        return _stitch_tiles(
            rows, [o // ratio for o in y_overlaps], [o // ratio for o in x_overlaps]
        )


# -------------------------------------------------------------------------------- decoder


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float, num_axes: int = 3):
        super().__init__()
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


def _rotate(hidden_states: torch.Tensor, rotary_emb) -> torch.Tensor:
    """Rotate-half rotary embedding on each head's leading channels."""
    cos, sin = (t.to(hidden_states.dtype) for t in rotary_emb)
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat([-second, first], dim=-1)
    return torch.cat([rotary * cos + rotated * sin, passthrough], dim=-1)


class _Attention(nn.Module):
    """Self-attention with FP32 Q/K RMS norms (no weights) and rotary embedding."""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads = heads
        inner_dim = heads * head_dim
        self.norm_q = nn.RMSNorm(head_dim, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim)])

    def forward(self, hidden_states: torch.Tensor, rotary_emb) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(2, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(2, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(2, (self.heads, -1))
        query = _rotate(self.norm_q(query.float()).to(query.dtype), rotary_emb)
        key = _rotate(self.norm_k(key.float()).to(key.dtype), rotary_emb)
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
        return self.to_out[0](hidden_states.flatten(2, 3))


class _Block(nn.Module):
    """Pre-norm attention and SwiGLU feed-forward, each scaled by a learned per-channel gain;
    the norms run in FP32."""

    def __init__(self, dim: int, heads: int, head_dim: int, ffn_mult: int, eps: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn = _Attention(dim, heads, head_dim, eps)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.ff = FeedForward(dim, mult=ffn_mult, activation_fn="swiglu", bias=True)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor, rotary_emb) -> torch.Tensor:
        normed = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = hidden_states + self.attn(normed, rotary_emb) * self.scale1
        normed = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        return hidden_states + self.ff(normed) * self.scale2


def _fp32_linear(linear: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    """``linear`` in its FP32 weights with autocast off, cast back to the input dtype."""
    with torch.autocast(hidden_states.device.type, enabled=False):
        output = linear(hidden_states.to(linear.weight.dtype))
    return output.to(hidden_states.dtype)


class _ViTDecoder(nn.Module):
    """Latent patches plus register tokens and a zero class token through the ViT blocks,
    then each patch back to ``patch_size_t x patch_size x patch_size`` pixels."""

    def __init__(self, c: SimpleNamespace):
        super().__init__()
        heads, head_dim = c.decoder_num_attention_heads, c.decoder_attention_head_dim
        dim = heads * head_dim
        self.patch_size = c.spatial_compression_ratio
        self.patch_size_t = c.temporal_compression_ratio
        self.out_channels = c.out_channels
        self.num_register_tokens = c.decoder_num_register_tokens
        self.rope = _RotaryEmbedding(int(head_dim * c.decoder_rope_dim_ratio), c.decoder_rope_theta)
        self.proj_in = nn.Linear(c.latent_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            _Block(dim, heads, head_dim, c.decoder_ffn_mult, c.decoder_norm_eps)
            for _ in range(c.decoder_num_layers)
        )
        self.norm_out = nn.LayerNorm(dim, eps=c.decoder_norm_eps)
        patch_pixels = self.patch_size_t * self.patch_size * self.patch_size
        self.proj_out = nn.Linear(dim, self.out_channels * patch_pixels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        dtype = hidden_states.dtype
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = _fp32_linear(self.proj_in, hidden_states)
        num_patches = hidden_states.shape[1]
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=dtype, device=hidden_states.device) / size) - 1.0
            for size in (num_frames, height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        cos, sin = self.rope(position_ids)
        rotary_emb = (cos.to(dtype), sin.to(dtype))
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = _fp32_linear(self.proj_out, self.norm_out(hidden_states))
        hidden_states = hidden_states[:, :num_patches, :]
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


class VideoDecoder(nn.Module):
    """A latent tile (on any device) to pixels on the decoder's device."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = _namespace(config)
        latent = self.config.latent_channels
        self.post_quant_conv = nn.Conv3d(latent, latent, 1)
        self.decoder = _ViTDecoder(self.config)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.post_quant_conv.weight.device)
        with torch.autocast(latents.device.type, dtype=torch.float16):
            return self.decoder(self.post_quant_conv(latents))


@torch.no_grad()
def decode_video(decoder: VideoDecoder, normalized_latents: torch.Tensor) -> torch.Tensor:
    """``(1, 3, F, H, W)`` pixels in [0, 1] on the latents' device, from normalized
    ``(1, 24, T, h, w)`` latents: the reference's overlapping temporal clips, each over
    256-pixel spatial tiles, every overlap blended linearly."""
    c = decoder.config
    temporal, spatial = c.temporal_compression_ratio, c.spatial_compression_ratio
    device = normalized_latents.device
    mean = torch.tensor(c.latents_mean, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(c.latents_std, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
    latents = normalized_latents.float() * std + mean
    tokens_chunk_size = math.ceil(c.clip_length / temporal)
    token_overlap = (-c.token_drop) % tokens_chunk_size
    frame_pre_padding = (-c.clip_length) % temporal
    frame_overlap = max(token_overlap * temporal - frame_pre_padding, 0)
    chunk_num_frames = tokens_chunk_size * temporal

    y_starts, y_lengths, y_overlaps = _split_tiles(latents.shape[-2] * spatial, alignment=spatial)
    x_starts, x_lengths, x_overlaps = _split_tiles(latents.shape[-1] * spatial, alignment=spatial)

    def decode_clip(clip: torch.Tensor) -> torch.Tensor:
        rows = []
        for y_start, y_length in zip(y_starts, y_lengths):
            y0, y1 = y_start // spatial, (y_start + y_length) // spatial
            row = []
            for x_start, x_length in zip(x_starts, x_lengths):
                x0, x1 = x_start // spatial, (x_start + x_length) // spatial
                row.append(decoder(clip[..., y0:y1, x0:x1]).float().to(device))
            rows.append(row)
        return _stitch_tiles(rows, y_overlaps, x_overlaps)

    num_tokens = latents.shape[2] + c.token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(c.token_drop > 0)
    if pad_tokens:
        latents = torch.cat([latents, latents[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    decoded_chunks: list[torch.Tensor] = []
    overlap = None
    for index in range(num_chunks):
        start = index * tokens_chunk_size
        clip = decode_clip(latents[:, :, start : start + tokens_chunk_size + token_overlap])
        for subchunk in range(int(c.token_drop > 0) + 1):
            frame_start = subchunk * chunk_num_frames
            chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
            chunk = chunk[:, :, frame_pre_padding:]
            if subchunk == 0:
                if overlap is not None:
                    chunk = _blend(overlap, chunk, frame_overlap, dim=-3)
                decoded_chunks.append(chunk)
            else:
                overlap = chunk
    if overlap is not None:
        decoded_chunks.append(overlap)
    video = torch.cat(decoded_chunks, dim=2)

    if pad_tokens:
        intra_tail = c.clip_length % temporal
        num_tokens_before_pad = latents.shape[2] - pad_tokens
        pad_frames = sum(
            (
                intra_tail
                if intra_tail and (num_tokens_before_pad + offset) % tokens_chunk_size == 0
                else temporal
            )
            for offset in range(pad_tokens)
        )
        video = video[:, :, :-pad_frames]

    pixel_mean = torch.tensor(_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    return (video * pixel_std + pixel_mean).clamp(0, 1)


# ---------------------------------------------------------------------------------- load


def load_fp32(module_cls, model_path, *, device) -> nn.Module:
    """``module_cls(config.json)`` with ``model_path``'s weights in FP32 on ``device``; the
    checkpoint's other tensors (the other half of the VAE) are left unread."""
    from accelerate import init_empty_weights

    config = json.loads((Path(model_path) / "config.json").read_text())
    with init_empty_weights(include_buffers=False):
        module = module_cls(config)
    materialize_meta_(module, device=device, dtype=torch.float32)
    load_checkpoint_into(module, str(model_path), tp_size=1, tp_rank=0, strict=True)
    return module.to(device).eval().requires_grad_(False)


def load_video_encoder(model_path, *, device) -> VideoEncoder:
    return load_fp32(VideoEncoder, model_path, device=device)


def load_video_decoder(model_path, *, device) -> VideoDecoder:
    return load_fp32(VideoDecoder, model_path, device=device)


__all__ = [
    "VideoDecoder",
    "VideoEncoder",
    "decode_video",
    "load_fp32",
    "load_video_decoder",
    "load_video_encoder",
]
