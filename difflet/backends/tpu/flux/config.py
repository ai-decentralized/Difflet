"""Geometry config for the FLUX.1-dev DiT on TPU (diffusers' transformer, no difflet fork)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

# black-forest-labs/FLUX.1-dev transformer/config.json (revision 3de623fc).
FLUX_1_DEV_CONFIG: dict[str, Any] = {
    "attention_head_dim": 128, "guidance_embeds": True, "in_channels": 64,
    "joint_attention_dim": 4096, "num_attention_heads": 24, "num_layers": 19,
    "num_single_layers": 38, "patch_size": 1, "pooled_projection_dim": 768,
    "axes_dims_rope": [16, 56, 56], "out_channels": None,
}


class TpuFluxConfig:
    """``transformer/config.json`` fields + runtime shape / topology."""

    def __init__(self, raw: dict[str, Any], *, height: int = 1024, width: int = 1024,
                 batch_size: int = 1, text_seq_len: int = 512, tp_degree: int = 1,
                 torch_dtype: torch.dtype = torch.bfloat16, vae_scale_factor: int = 8,
                 context_parallel_enabled: bool = False, cfg_parallel_enabled: bool = False,
                 sp_enabled: bool = False) -> None:
        for key, value in raw.items():
            if not key.startswith("_"):
                setattr(self, key, value)
        if getattr(self, "out_channels", None) is None:
            self.out_channels = self.in_channels
        if isinstance(getattr(self, "axes_dims_rope", None), list):
            self.axes_dims_rope = tuple(self.axes_dims_rope)
        self.height, self.width = int(height), int(width)
        self.batch_size = int(batch_size)
        self.text_seq_len = int(text_seq_len)
        self.tp_degree = int(tp_degree)
        self.torch_dtype = torch_dtype
        self.vae_scale_factor = int(vae_scale_factor)
        self.context_parallel_enabled = bool(context_parallel_enabled)
        self.cfg_parallel_enabled = bool(cfg_parallel_enabled)
        self.sp_enabled = bool(sp_enabled)
        self.neuron_config = SimpleNamespace(batch_size=self.batch_size, torch_dtype=torch_dtype,
                                             tp_degree=self.tp_degree)

    @property
    def inner_dim(self) -> int:
        return int(self.num_attention_heads) * int(self.attention_head_dim)

    @property
    def latent_height(self) -> int:
        # The VAE downsamples 8x; packing pairs 2x2 latent pixels into one token.
        return 2 * (int(self.height) // (self.vae_scale_factor * 2))

    @property
    def latent_width(self) -> int:
        return 2 * (int(self.width) // (self.vae_scale_factor * 2))

    @property
    def image_seq_len(self) -> int:
        return (self.latent_height // 2) * (self.latent_width // 2)

    def validate(self) -> None:
        tp = int(self.tp_degree)
        if int(self.num_attention_heads) % tp:
            raise ValueError(f"Flux has {self.num_attention_heads} heads, which does not divide tp={tp}")
        if (int(self.inner_dim) * 4) % tp:
            raise ValueError(f"Flux mlp dim {self.inner_dim * 4} does not divide tp={tp}")
        for name in ("height", "width"):
            if int(getattr(self, name)) % (self.vae_scale_factor * 2):
                raise ValueError(
                    f"Flux {name}={getattr(self, name)} must be divisible by "
                    f"{self.vae_scale_factor * 2} (vae {self.vae_scale_factor}x, 2x2 packing)"
                )
        for mode, enabled in (("context parallelism", self.context_parallel_enabled),
                              ("cfg parallelism", self.cfg_parallel_enabled),
                              ("sequence parallelism", self.sp_enabled)):
            if enabled:
                raise NotImplementedError(f"the TPU backend does not implement {mode} yet; run with tp only")

    @classmethod
    def from_pretrained(cls, transformer_path, **overrides) -> "TpuFluxConfig":
        raw = json.loads((Path(transformer_path) / "config.json").read_text())
        config = cls(raw, **overrides)
        config.validate()
        return config


__all__ = ["FLUX_1_DEV_CONFIG", "TpuFluxConfig"]
