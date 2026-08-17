"""Geometry config for the Qwen-Image transformer on TPU.

The Trainium side derives this from ``InferenceConfig``/``NeuronConfig``,
which are NxD types. The geometry itself is not Trainium-specific — it is
patch/VAE arithmetic — so this recomputes the same properties from the
diffusers ``config.json`` without dragging in NxD.

Kept deliberately as a plain dataclass rather than a shared base with the
Trainium config: the two backends need the same *numbers*, not a coupled
class hierarchy, and duplicating ~40 lines of arithmetic is cheaper than a
refactor of the NxD config on hardware we cannot run here.

Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class TpuQwenImageConfig:
    """Duck-typed config for ``_QwenImageTransformerTraceModule``.

    Attribute names match what the backend-neutral modeling reads, so the same
    modeling code runs on either backend.
    """

    # --- diffusers geometry ---
    patch_size: int
    in_channels: int
    out_channels: int
    num_layers: int
    attention_head_dim: int
    num_attention_heads: int
    joint_attention_dim: int
    guidance_embeds: bool
    axes_dims_rope: tuple[int, ...]

    # --- difflet runtime shape ---
    height: int = 1024
    width: int = 1024
    text_seq_len: int = 1024
    batch_size: int = 1
    vae_scale_factor: int = 8
    tp_degree: int = 1
    dp_degree: int = 1
    context_parallel_enabled: bool = False
    cp_mode: str = "gather_kv"
    zero_cond_t: bool = False
    use_layer3d_rope: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    extras: dict = field(default_factory=dict)

    # --- derived geometry (mirrors QwenImageTransformerInferenceConfig) ---
    @property
    def latent_height(self) -> int:
        return 2 * (int(self.height) // (int(self.vae_scale_factor) * 2))

    @property
    def latent_width(self) -> int:
        return 2 * (int(self.width) // (int(self.vae_scale_factor) * 2))

    @property
    def packed_height(self) -> int:
        return self.latent_height // int(self.patch_size)

    @property
    def packed_width(self) -> int:
        return self.latent_width // int(self.patch_size)

    @property
    def image_seq_len(self) -> int:
        return self.packed_height * self.packed_width

    def validate(self) -> None:
        if sum(int(d) for d in self.axes_dims_rope) != int(self.attention_head_dim):
            raise ValueError("Qwen-Image axes_dims_rope must sum to attention_head_dim")
        if any(int(d) % 2 for d in self.axes_dims_rope):
            raise ValueError("Qwen-Image axes_dims_rope entries must be even")
        if int(self.patch_size) != 2:
            raise NotImplementedError("Qwen-Image currently supports patch_size=2")
        for name in ("height", "width"):
            value = int(getattr(self, name))
            if value % (int(self.vae_scale_factor) * 2):
                raise ValueError(f"Qwen-Image compile {name} must be divisible by 16")
        # tp shards the attention heads, so it has to divide them evenly. This
        # is the check that rules out the tiny CI fixture (3 heads) at tp=4.
        if int(self.num_attention_heads) % int(self.tp_degree):
            raise ValueError(
                f"Qwen-Image has {self.num_attention_heads} attention heads, "
                f"which does not divide tp={self.tp_degree}"
            )

    @classmethod
    def from_pretrained(cls, transformer_path, **overrides) -> "TpuQwenImageConfig":
        """Build from a diffusers ``transformer/config.json``."""
        body = json.loads((Path(transformer_path) / "config.json").read_text())
        known = {
            "patch_size", "in_channels", "out_channels", "num_layers",
            "attention_head_dim", "num_attention_heads", "joint_attention_dim",
            "guidance_embeds",
        }
        kwargs = {k: body[k] for k in known if k in body}
        kwargs["axes_dims_rope"] = tuple(body["axes_dims_rope"])
        kwargs.setdefault("out_channels", int(body["in_channels"]) // 4)
        kwargs.update(overrides)
        config = cls(**kwargs)
        config.validate()
        return config


__all__ = ["TpuQwenImageConfig"]
