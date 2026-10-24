"""Neuron application for HunyuanVideo's VAE decoder: diffusers' AutoencoderKLHunyuanVideo.

Decodes as HunyuanVideoPipeline does, ``vae.decode(latents / scaling_factor)``, eager on one core
(the separate VAE stage), in fp32 by default (the checkpoint is fp32, and Wan and Flux decode in
fp32 too); bf16 is accepted. The encoder is dropped after loading.

Tiling is always on, in space and in time: diffusers decodes ``tile_frames``-frame windows of the
clip one after another (blending the overlap) and splits each window into ``tile_size``-pixel
tiles, so a tile's activations do not grow with the clip length (HunyuanVideo 1.5's autoencoder
tiles space only, which is why its long clips ran out of memory). Both strides are 3/4 of the tile
(25% overlap, diffusers' default ratio; its defaults are 256 pixels and 16 frames). diffusers walks
the latent by the stride in latent units and crops the decoded tiles by the stride in sample
units, so both must be whole latent units: ``tile_size`` a multiple of 32 pixels and
``tile_frames`` a multiple of 16 frames at the 8x spatial and 4x temporal compression; anything
else is refused rather than stitched off by a fraction of a latent.

The config is read from ``config.json`` at construction, so ``latent_shape`` and
``scaling_factor`` are known before the weights load.
"""

from __future__ import annotations

import json
import os
import time

import torch


class NeuronHunyuanVideoVaeDecoderApplication:
    """``load(skip_warmup=False)``, then ``decode(latents) -> video`` ([B, 3, F, H, W], in [-1, 1]).

    ``model_path`` is the VAE's directory (``<model>/vae``); ``height``, ``width`` and
    ``num_frames`` are the sample shape the warm-up decodes.
    """

    def __init__(self, *, model_path, height: int = 320, width: int = 512, num_frames: int = 61,
                 tile_size: int = 256, tile_frames: int = 16, dtype=torch.float32,
                 device="neuron") -> None:
        self.model_path = str(model_path)
        self.height, self.width, self.num_frames = int(height), int(width), int(num_frames)
        self.tile_size, self.tile_frames = int(tile_size), int(tile_frames)
        self.dtype, self.device = dtype, torch.device(device)
        if self.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"the HunyuanVideo VAE decoder runs in bf16 or fp32, got {self.dtype}")
        with open(os.path.join(self.model_path, "config.json"), encoding="utf-8") as f:
            self.vae_config = json.load(f)
        self._check_grid()
        self.vae = None
        self.phase_seconds: dict[str, float] = {}

    @property
    def _compression(self) -> tuple[int, int]:
        cfg = self.vae_config
        return int(cfg.get("spatial_compression_ratio", 8)), int(cfg.get("temporal_compression_ratio", 4))

    def _check_grid(self) -> None:
        s, t = self._compression
        for name, n in (("height", self.height), ("width", self.width)):
            if n % s:
                raise ValueError(f"HunyuanVideo VAE {name}={n} must be a multiple of {s}")
        if self.num_frames % t != 1:
            raise ValueError(f"HunyuanVideo VAE num_frames={self.num_frames} must be {t}k+1")
        for name, n, step in (("tile_size", self.tile_size, s), ("tile_frames", self.tile_frames, t)):
            if n % step or (n * 3 // 4) % step:
                raise ValueError(
                    f"HunyuanVideo VAE {name}={n}: the tile and its stride ({n * 3 // 4}) must be "
                    f"whole latent units of {step} (a multiple of {4 * step})"
                )

    @property
    def scaling_factor(self) -> float:
        return float(self.vae_config.get("scaling_factor", 0.476986))

    @property
    def latent_shape(self) -> tuple[int, ...]:
        s, t = self._compression
        channels = int(self.vae_config.get("latent_channels", 16))
        return (1, channels, (self.num_frames - 1) // t + 1, self.height // s, self.width // s)

    def load(self, skip_warmup: bool = False) -> None:
        from diffusers import AutoencoderKLHunyuanVideo

        t = time.perf_counter()
        vae = AutoencoderKLHunyuanVideo.from_pretrained(self.model_path, torch_dtype=self.dtype).eval()
        vae.encoder = None  # text-to-video only decodes
        stride, stride_frames = self.tile_size * 3 // 4, self.tile_frames * 3 // 4
        vae.enable_tiling(tile_sample_min_height=self.tile_size, tile_sample_min_width=self.tile_size,
                          tile_sample_min_num_frames=self.tile_frames,
                          tile_sample_stride_height=stride, tile_sample_stride_width=stride,
                          tile_sample_stride_num_frames=stride_frames)
        self.vae = vae.to(self.device).requires_grad_(False)
        self.phase_seconds["load"] = time.perf_counter() - t
        if not skip_warmup:  # compiles the eager kernels for this shape (once; cached)
            t = time.perf_counter()
            self.decode(torch.zeros(self.latent_shape, dtype=self.dtype))
            self.phase_seconds["warmup decode"] = time.perf_counter() - t

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(self.device, self.dtype) / self.scaling_factor
        return self.vae.decode(z, return_dict=False)[0]


__all__ = ["NeuronHunyuanVideoVaeDecoderApplication"]
