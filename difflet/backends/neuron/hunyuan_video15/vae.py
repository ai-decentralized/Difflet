"""Neuron application for HunyuanVideo 1.5's VAE decoder: diffusers' AutoencoderKLHunyuanVideo15.

Decodes as HunyuanVideo15Pipeline does, ``vae.decode(latents / scaling_factor)``, eager on one
core (the separate VAE stage), in bf16 by default (on the device bf16 matched the CPU's tiled fp32
decode to 4.3e-3; fp32 to 3.3e-7). The encoder is dropped after loading.

Tiling is always on: diffusers tiles space only, so a tile's activations grow with the clip length
(at 61 frames, 256 x 256-pixel fp32 tiles needed 2 GB tensors). The tile size is in sample pixels
(25% overlap, diffusers' default); smaller tiles fit longer clips but blend more seams (against the
untiled decode, 5 frames: 1.2e-2 at 256 pixels, 3.2e-2 at 128). The default is 128: at 61 frames 256
pixels runs out of memory even in bf16, 128 fits (44.7 s, 6.3 GiB peak); temporal chunking would
lift this. Both of diffusers' tile attributes
are set: its tiling loop reads the latent-unit sizes, which ``enable_tiling`` does not derive from
the sample-pixel ones.
"""

from __future__ import annotations

import os
import time

import torch


class NeuronHunyuanVideo15VaeDecoderApplication:
    """``load(skip_warmup=False)``, then ``decode(latents) -> video`` ([B, 3, F, H, W], in [-1, 1])."""

    def __init__(self, *, model_path, height: int = 480, width: int = 848, num_frames: int = 61,
                 tile_size: int = 128, dtype=torch.bfloat16, device="neuron") -> None:
        self.model_path = str(model_path)
        self.height, self.width, self.num_frames = int(height), int(width), int(num_frames)
        self.tile_size, self.dtype, self.device = int(tile_size), dtype, torch.device(device)
        self.vae = None
        self.phase_seconds: dict[str, float] = {}

    @property
    def scaling_factor(self) -> float:
        return float(self.vae.config.scaling_factor)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        cfg = self.vae.config
        s, t = int(cfg.spatial_compression_ratio), int(cfg.temporal_compression_ratio)
        return (1, int(cfg.latent_channels), (self.num_frames - 1) // t + 1, self.height // s, self.width // s)

    def load(self, skip_warmup: bool = False) -> None:
        from diffusers import AutoencoderKLHunyuanVideo15

        t = time.perf_counter()
        vae = AutoencoderKLHunyuanVideo15.from_pretrained(self.model_path, torch_dtype=self.dtype).eval()
        vae.encoder = None  # text-to-video only decodes
        s = int(vae.config.spatial_compression_ratio)
        vae.enable_tiling()
        vae.tile_sample_min_height = vae.tile_sample_min_width = self.tile_size
        vae.tile_latent_min_height = vae.tile_latent_min_width = self.tile_size // s
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


__all__ = ["NeuronHunyuanVideo15VaeDecoderApplication"]
