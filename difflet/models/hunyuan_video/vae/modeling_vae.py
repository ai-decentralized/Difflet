"""HunyuanVideo VAE decoder modeling.

Decoder-only wrapper for Trainium inference: ``post_quant_conv`` followed by
the diffusers HunyuanVideo 3D decoder. Encoder, KL sampling, and host-side
tiling remain outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HunyuanVideoVAEDecoderConfig:
    """Config subset needed for HunyuanVideo VAE decoding."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    up_block_types: tuple[str, ...] = (
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
    )
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    norm_num_groups: int = 32
    scaling_factor: float = 0.476986
    spatial_compression_ratio: int = 8
    temporal_compression_ratio: int = 4
    mid_block_add_attention: bool = True
    tile_sample_min_height: int = 256
    tile_sample_min_width: int = 256
    tile_sample_min_num_frames: int = 16
    tile_sample_stride_height: int = 192
    tile_sample_stride_width: int = 192
    tile_sample_stride_num_frames: int = 12
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.up_block_types = tuple(self.up_block_types)
        self.block_out_channels = tuple(int(x) for x in self.block_out_channels)
        if any(block != "HunyuanVideoUpBlock3D" for block in self.up_block_types):
            raise NotImplementedError("HunyuanVideo VAE supports only HunyuanVideoUpBlock3D.")
        if int(self.spatial_compression_ratio) != 8:
            raise NotImplementedError("HunyuanVideo VAE spike expects spatial compression ratio 8.")
        if int(self.temporal_compression_ratio) != 4:
            raise NotImplementedError("HunyuanVideo VAE spike expects temporal compression ratio 4.")
        if len(self.up_block_types) != len(self.block_out_channels):
            raise ValueError("up_block_types and block_out_channels must have the same length.")

    @classmethod
    def from_diffusers_dict(cls, raw: dict[str, Any]) -> "HunyuanVideoVAEDecoderConfig":
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        kept = {key: value for key, value in raw.items() if key in fields}
        extras = {key: value for key, value in raw.items() if key not in fields}
        kept["extra_config"] = extras
        return cls(**kept)

    @property
    def tile_latent_height(self) -> int:
        return int(self.tile_sample_min_height) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_width(self) -> int:
        return int(self.tile_sample_min_width) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_frames(self) -> int:
        return int(self.tile_sample_min_num_frames) // int(self.temporal_compression_ratio) + 1


class HunyuanVideoVAEDecoderModel(nn.Module):
    """Decoder-only HunyuanVideo VAE module with HF-compatible key names."""

    def __init__(self, config: HunyuanVideoVAEDecoderConfig) -> None:
        super().__init__()
        from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import (
            HunyuanVideoDecoder3D,
        )

        self.config = config
        self.post_quant_conv = nn.Conv3d(
            int(config.latent_channels),
            int(config.latent_channels),
            kernel_size=1,
        )
        self.decoder = HunyuanVideoDecoder3D(
            in_channels=int(config.latent_channels),
            out_channels=int(config.out_channels),
            up_block_types=tuple(config.up_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=int(config.layers_per_block),
            norm_num_groups=int(config.norm_num_groups),
            act_fn=str(config.act_fn),
            time_compression_ratio=int(config.temporal_compression_ratio),
            spatial_compression_ratio=int(config.spatial_compression_ratio),
            mid_block_add_attention=bool(config.mid_block_add_attention),
        )
        _replace_interpolate_upsamplers(self.decoder)
        _replace_group_norms_with_fp32(self.decoder)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self.post_quant_conv(latents)
        return self.decoder(latents)


class _RepeatNearestUpsampleCausal3D(nn.Module):
    """Nearest upsample equivalent that avoids ``F.interpolate`` in XLA trace."""

    def __init__(self, conv: nn.Module, upsample_factor: tuple[float, float, float]) -> None:
        super().__init__()
        self.conv = conv
        self.upsample_factor = tuple(int(factor) for factor in upsample_factor)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = _repeat_nearest_2d(hidden_states, self.upsample_factor[1:])
        if self.upsample_factor[0] != 1:
            hidden_states = _repeat_causal_time(hidden_states, self.upsample_factor[0])
        return self.conv(hidden_states)


def _repeat_nearest_2d(x: torch.Tensor, factors: tuple[int, int]) -> torch.Tensor:
    if factors[0] != 1:
        x = x.repeat_interleave(factors[0], dim=3)
    if factors[1] != 1:
        x = x.repeat_interleave(factors[1], dim=4)
    return x


def _repeat_nearest_3d(x: torch.Tensor, factors: tuple[int, int, int]) -> torch.Tensor:
    if factors[0] != 1:
        x = x.repeat_interleave(factors[0], dim=2)
    if factors[1] != 1:
        x = x.repeat_interleave(factors[1], dim=3)
    if factors[2] != 1:
        x = x.repeat_interleave(factors[2], dim=4)
    return x


def _repeat_causal_time(x: torch.Tensor, factor: int) -> torch.Tensor:
    num_frames = x.size(2)
    if factor == 1 or num_frames <= 1:
        return x
    chunks = [x[:, :, :1]]
    for frame in range(1, num_frames):
        frame_slice = x[:, :, frame : frame + 1]
        chunks.extend([frame_slice] * factor)
    return torch.cat(chunks, dim=2)


class _Fp32GroupNorm(nn.GroupNorm):
    """``GroupNorm`` whose reduction runs in fp32.

    The decoder normalizes over group volumes of several million elements
    (4.5M per group at the 256x256x17 tile). A bf16 accumulator saturates well
    before that, so the on-device statistics come out wrong by far more than
    bf16 rounding would explain. Subclassing keeps ``weight``/``bias`` names
    intact, so checkpoint keys are unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.group_norm(x.float(), self.num_groups, weight, bias, self.eps).to(out_dtype)


def _replace_group_norms_with_fp32(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm) and not isinstance(child, _Fp32GroupNorm):
            replacement = _Fp32GroupNorm(
                child.num_groups,
                child.num_channels,
                eps=child.eps,
                affine=child.affine,
            )
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            _replace_group_norms_with_fp32(child)


def _replace_interpolate_upsamplers(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "HunyuanVideoUpsampleCausal3D":
            setattr(
                module,
                name,
                _RepeatNearestUpsampleCausal3D(
                    conv=child.conv,
                    upsample_factor=tuple(child.upsample_factor),
                ),
            )
        else:
            _replace_interpolate_upsamplers(child)
