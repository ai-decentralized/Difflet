"""Geometry config for the HunyuanVideo DiT on TPU.

Like Wan, HunyuanVideo's backend-neutral modeling ships its own hyperparameter
dataclass (``HunyuanVideoTransformerConfig``); this extends it with the
runtime shape, the tp degree and the dtype the TPU component lifecycle needs,
and derives the latent geometry the example inputs are built from.

Third model on the TPU backend, after Qwen-Image and Wan; see
docs/plans/2026-09-11-tpu-port-hunyuan-ltx2-flux.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from difflet.models.hunyuan_video.modeling_hunyuan_video import (
    HunyuanVideoTransformerConfig,
)


@dataclass
class TpuHunyuanVideoConfig(HunyuanVideoTransformerConfig):
    """``HunyuanVideoTransformerConfig`` plus the TPU runtime's shape and topology."""

    # --- difflet runtime shape (registry default 320x512x61) ---
    height: int = 320
    width: int = 512
    num_frames: int = 61
    batch_size: int = 1
    # The serving adapter's Llama bucket minus the template crop (256 tokens).
    text_seq_len: int = 256
    # AutoencoderKLHunyuanVideo: 8x spatial, 4x temporal (causal, 4n+1 frames).
    vae_scale_factor_spatial: int = 8
    vae_scale_factor_temporal: int = 4

    # --- topology ---
    tp_degree: int = 1
    torch_dtype: torch.dtype = torch.bfloat16
    cfg_parallel_enabled: bool = False

    # --- derived latent geometry ---
    @property
    def latent_frames(self) -> int:
        return (int(self.num_frames) - 1) // int(self.vae_scale_factor_temporal) + 1

    @property
    def latent_height(self) -> int:
        return int(self.height) // int(self.vae_scale_factor_spatial)

    @property
    def latent_width(self) -> int:
        return int(self.width) // int(self.vae_scale_factor_spatial)

    @property
    def image_seq_len(self) -> int:
        return (
            (self.latent_frames // int(self.patch_size_t))
            * (self.latent_height // int(self.patch_size))
            * (self.latent_width // int(self.patch_size))
        )

    def validate(self) -> None:
        tp = int(self.tp_degree)
        if int(self.num_attention_heads) % tp:
            raise ValueError(
                f"HunyuanVideo has {self.num_attention_heads} attention heads, "
                f"which does not divide tp={tp}"
            )
        mlp_dim = int(self.inner_dim * float(self.mlp_ratio))
        if mlp_dim % tp:
            raise ValueError(f"HunyuanVideo mlp dim {mlp_dim} does not divide tp={tp}")
        spatial = int(self.vae_scale_factor_spatial)
        patch = int(self.patch_size)
        # Check the requested pixel size, not just the latent grid: the latent
        # dims are floor divisions, so an unaligned size would silently
        # generate a different shape than the caller asked for.
        for name in ("height", "width"):
            value = int(getattr(self, name))
            if value % (spatial * patch):
                raise ValueError(
                    f"HunyuanVideo {name}={value} must be divisible by "
                    f"{spatial * patch} (vae {spatial}x, patch {patch})"
                )
        temporal = int(self.vae_scale_factor_temporal)
        if (int(self.num_frames) - 1) % temporal:
            raise ValueError(
                f"HunyuanVideo num_frames={self.num_frames} must satisfy "
                f"(num_frames - 1) % {temporal} == 0"
            )
        if self.latent_frames % int(self.patch_size_t):
            raise ValueError(
                f"HunyuanVideo latent frames {self.latent_frames} is not divisible "
                f"by the temporal patch {self.patch_size_t}"
            )
        for mode, enabled in (
            ("context parallelism", self.context_parallel_enabled),
            ("cfg parallelism", self.cfg_parallel_enabled),
            ("sequence parallelism", self.sp_enabled),
        ):
            if enabled:
                raise NotImplementedError(
                    f"the TPU backend does not implement {mode} yet; run with tp only"
                )

    @classmethod
    def from_pretrained(cls, transformer_path, **overrides) -> "TpuHunyuanVideoConfig":
        """Build from a diffusers ``transformer/config.json``."""
        raw = json.loads((Path(transformer_path) / "config.json").read_text())
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kept = {k: v for k, v in raw.items() if k in fields}
        if isinstance(kept.get("rope_axes_dim"), list):
            kept["rope_axes_dim"] = tuple(kept["rope_axes_dim"])
        kept.update(overrides)
        config = cls(**kept)
        config.validate()
        return config


__all__ = ["TpuHunyuanVideoConfig"]
