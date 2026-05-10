"""Wan VAE decoder modeling for the M2 spike.

This module is hardware-neutral: it imports only torch, stdlib, and pure
diffusers utilities. W3c intentionally implements decoder-only inference:
``post_quant_conv`` followed by ``WanDecoder3d`` with causal-conv feature
cache. Encoder, KL sampling, and tiled decode remain outside the spike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.activations import get_activation

CACHE_T = 2


def _tail_cache(x: torch.Tensor) -> torch.Tensor:
    start = max(int(x.shape[2]) - CACHE_T, 0)
    return x[:, :, start:, :, :].clone()


@dataclass
class WanVAEDecoderConfig:
    """Config subset needed for Wan VAE decoding."""

    base_dim: int = 96
    decoder_base_dim: Optional[int] = None
    z_dim: int = 16
    dim_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    attn_scales: list[float] = field(default_factory=list)
    temperal_downsample: list[bool] = field(default_factory=lambda: [False, True, True])
    dropout: float = 0.0
    latents_mean: list[float] = field(default_factory=list)
    latents_std: list[float] = field(default_factory=list)
    is_residual: bool = False
    in_channels: int = 3
    out_channels: int = 3
    patch_size: Optional[int] = None
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8

    def __post_init__(self) -> None:
        if self.decoder_base_dim is None:
            self.decoder_base_dim = self.base_dim
        if self.is_residual:
            raise NotImplementedError("Wan VAE spike supports only is_residual=False.")
        if self.patch_size is not None:
            raise NotImplementedError("Wan VAE spike does not support patchified VAE.")
        if self.scale_factor_temporal != 4:
            raise NotImplementedError("Wan VAE spike expects temporal scale factor 4.")
        if self.scale_factor_spatial != 8:
            raise NotImplementedError("Wan VAE spike expects spatial scale factor 8.")
        if len(self.dim_mult) < 2:
            raise ValueError("dim_mult must contain at least two stages.")
        if len(self.temperal_downsample) != len(self.dim_mult) - 1:
            raise ValueError("temperal_downsample must have len(dim_mult) - 1 entries.")

    @property
    def temperal_upsample(self) -> list[bool]:
        return list(self.temperal_downsample[::-1])

    @classmethod
    def from_diffusers_dict(cls, raw: dict) -> "WanVAEDecoderConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kept = {k: v for k, v in raw.items() if k in fields}
        return cls(**kept)


class WanCausalConv3d(nn.Conv3d):
    """Conv3d with causal padding in the temporal dimension."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class WanRMSNorm(nn.Module):
    """Channel-first RMS norm used by Wan VAE blocks."""

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        needs_fp32 = x.dtype in (torch.float16, torch.bfloat16) or any(
            marker in str(x.dtype) for marker in ("float4_", "float8_")
        )
        normalized = F.normalize(
            x.float() if needs_fp32 else x, dim=(1 if self.channel_first else -1)
        ).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class WanUpsample(nn.Upsample):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class WanResample(nn.Module):
    """Spatial and temporal upsampling used by the Wan decoder."""

    def __init__(self, dim: int, mode: str, upsample_out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "none":
            self.resample = nn.Identity()
        else:
            raise NotImplementedError(f"unsupported Wan VAE resample mode: {mode}")

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list[Optional[torch.Tensor] | str]] = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d" and feat_cache is not None:
            assert feat_idx is not None
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = _tail_cache(x)
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                        dim=2,
                    )
                if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        return x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)


class WanResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.nonlinearity = get_activation(non_linearity)
        self.norm1 = WanRMSNorm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list[Optional[torch.Tensor] | str]] = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        h = self.conv_shortcut(x)
        x = self.nonlinearity(self.norm1(x))

        if feat_cache is not None:
            assert feat_idx is not None
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.dropout(self.nonlinearity(self.norm2(x)))
        if feat_cache is not None:
            assert feat_idx is not None
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)
        return x + h


class WanAttentionBlock(nn.Module):
    """Single-head spatial attention per frame."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = WanRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch_size, channels, time, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)
        x = x.view(batch_size, time, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + identity


class WanMidBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list[Optional[torch.Tensor] | str]] = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attn(x), feat_cache=feat_cache, feat_idx=feat_idx)
        return x


class WanUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        current_dim = in_dim
        self.resnets = nn.ModuleList()
        for _ in range(num_res_blocks + 1):
            self.resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.upsamplers = (
            nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])
            if upsample_mode is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list[Optional[torch.Tensor] | str]] = None,
        feat_idx: list[int] | None = None,
        first_chunk: Optional[bool] = None,
    ) -> torch.Tensor:
        del first_chunk
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x


class WanDecoder3d(nn.Module):
    """Wan 3D VAE decoder body."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temperal_upsample: list[bool] | None = None,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        dim_mult = dim_mult if dim_mult is not None else [1, 2, 4, 4]
        attn_scales = attn_scales if attn_scales is not None else []
        temperal_upsample = temperal_upsample if temperal_upsample is not None else [True, True, False]
        if attn_scales:
            raise NotImplementedError("Wan VAE spike does not support decoder attention scales yet.")

        self.nonlinearity = get_activation(non_linearity)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)
        self.up_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2
            up_flag = i != len(dim_mult) - 1
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            else:
                upsample_mode = None
            self.up_blocks.append(
                WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            )
        self.norm_out = WanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list[Optional[torch.Tensor] | str]] = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        del first_chunk
        if feat_cache is not None:
            assert feat_idx is not None
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.nonlinearity(self.norm_out(x))

        if feat_cache is not None:
            assert feat_idx is not None
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class WanVAEDecoderModel(nn.Module):
    """Decoder-only Wan VAE component.

    Input is denoised latent video ``(B, z_dim, T_latent, H/8, W/8)``.
    Output is decoded video tensor ``(B, out_channels, T_video, H, W)``.
    """

    def __init__(self, config: WanVAEDecoderConfig):
        super().__init__()
        self.config = config
        self.post_quant_conv = WanCausalConv3d(config.z_dim, config.z_dim, 1)
        self.decoder = WanDecoder3d(
            dim=int(config.decoder_base_dim),
            z_dim=config.z_dim,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            attn_scales=config.attn_scales,
            temperal_upsample=config.temperal_upsample,
            dropout=config.dropout,
            out_channels=config.out_channels,
        )
        self._conv_count = sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())

    def _clear_cache(self) -> list[Optional[torch.Tensor] | str]:
        return [None] * self._conv_count

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, _, _ = z.shape
        feat_cache = self._clear_cache()
        z = self.post_quant_conv(z)
        chunks = []
        for i in range(num_frames):
            conv_idx = [0]
            chunk = self.decoder(
                z[:, :, i : i + 1, :, :],
                feat_cache=feat_cache,
                feat_idx=conv_idx,
                first_chunk=i == 0,
            )
            chunks.append(chunk)
        return torch.clamp(torch.cat(chunks, dim=2), min=-1.0, max=1.0)


__all__ = [
    "WanCausalConv3d",
    "WanDecoder3d",
    "WanRMSNorm",
    "WanVAEDecoderConfig",
    "WanVAEDecoderModel",
]
