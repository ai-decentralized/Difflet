"""Geometry config for the Flux DiT, backend-neutral.

The Trainium side builds ``FluxBackboneInferenceConfig`` (an NxD ``InferenceConfig``,
in ``difflet.models.flux.trainium_backbone``). Off Trainium the same
``NeuronFluxTransformer2DModel`` runs from this plain dataclass: its attribute names
are exactly what the modeling reads, including ``neuron_config.torch_dtype`` and
``neuron_config.world_size`` (the modeling keeps reading
``config.neuron_config.*`` so the Trainium path is untouched). Modelled on
``difflet.models.qwen_image.dit_config.QwenImageDiTConfig``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch


@dataclass
class FluxDiTConfig:
    """Duck-typed config for ``NeuronFluxTransformer2DModel``."""

    # --- diffusers geometry ---
    patch_size: int
    in_channels: int
    out_channels: int | None  # None -> the modeling uses in_channels (diffusers' rule)
    num_layers: int
    num_single_layers: int
    attention_head_dim: int
    num_attention_heads: int
    joint_attention_dim: int
    pooled_projection_dim: int
    guidance_embeds: bool
    axes_dims_rope: tuple[int, ...] = (16, 56, 56)

    # --- difflet runtime ---
    height: int = 1024
    width: int = 1024
    tp_degree: int = 1
    world_size: int = 1
    torch_dtype: torch.dtype = torch.bfloat16
    vae_scale_factor: int = 8
    cfg_parallel_enabled: bool = False
    context_parallel_enabled: bool = False
    cp_mode: str = "gather_kv"
    sp_enabled: bool = False

    @property
    def inner_dim(self) -> int:
        return int(self.num_attention_heads) * int(self.attention_head_dim)

    @property
    def image_seq_len(self) -> int:
        """Packed image tokens: one per 2x2 patch of the VAE latent."""
        return int(self.height) * int(self.width) // (2 * int(self.vae_scale_factor)) ** 2

    @property
    def neuron_config(self) -> SimpleNamespace:
        """The ``config.neuron_config.*`` names the modeling reads."""
        return SimpleNamespace(
            torch_dtype=self.torch_dtype, world_size=self.world_size, tp_degree=self.tp_degree
        )

    @classmethod
    def from_pretrained(cls, transformer_dir, **overrides) -> "FluxDiTConfig":
        """Build from a diffusers ``transformer/config.json``."""
        body = json.loads((Path(transformer_dir) / "config.json").read_text())
        known = (
            "patch_size",
            "in_channels",
            "num_layers",
            "num_single_layers",
            "attention_head_dim",
            "num_attention_heads",
            "joint_attention_dim",
            "pooled_projection_dim",
            "guidance_embeds",
        )
        kwargs = {key: body[key] for key in known}
        kwargs["out_channels"] = body.get("out_channels")
        if "axes_dims_rope" in body:
            kwargs["axes_dims_rope"] = tuple(body["axes_dims_rope"])
        kwargs.update(overrides)
        return cls(**kwargs)


__all__ = ["FluxDiTConfig"]
