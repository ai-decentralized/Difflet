"""Backend-neutral HunyuanVideo geometry for the eager GPU component."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from difflet.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoTransformerConfig


@dataclass
class HunyuanVideoDiTConfig(HunyuanVideoTransformerConfig):
    joint_attention_mask_mode: str = "compact"
    height: int = 320
    width: int = 512
    num_frames: int = 61
    text_seq_len: int = 256
    tp_degree: int = 1
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        super().__post_init__()
        if self.image_condition_type is not None or not self.guidance_embeds:
            raise NotImplementedError("CUDA HunyuanVideo supports guidance-distilled text-to-video")
        if self.context_parallel_enabled or self.sp_enabled or self.tp_degree != 1:
            raise NotImplementedError("CUDA HunyuanVideo requires tp=1, cp=1 and no SP")
        if self.height <= 0 or self.width <= 0 or self.height % 16 or self.width % 16:
            raise ValueError("HunyuanVideo height and width must be positive multiples of 16")
        if self.num_frames <= 0 or (self.num_frames - 1) % 4:
            raise ValueError("HunyuanVideo num_frames must be positive and 4n+1")
        if self.text_seq_len <= 0:
            raise ValueError("HunyuanVideo text_seq_len must be positive")
        if self.patch_size != 2 or self.patch_size_t != 1:
            raise NotImplementedError("CUDA HunyuanVideo requires patch_size=2 and patch_size_t=1")
        if sum(self.rope_axes_dim) != self.attention_head_dim or any(
            x < 0 or x % 2 for x in self.rope_axes_dim
        ):
            raise ValueError("HunyuanVideo RoPE axes must be even and sum to attention_head_dim")

    @property
    def latent_height(self):
        return self.height // 8

    @property
    def latent_width(self):
        return self.width // 8

    @property
    def latent_frames(self):
        return (self.num_frames - 1) // 4 + 1

    @classmethod
    def from_pretrained(cls, transformer_path, **overrides):
        body = json.loads((Path(transformer_path) / "config.json").read_text())
        name = body.get("_class_name", "HunyuanVideoTransformer3DModel")
        if name != "HunyuanVideoTransformer3DModel":
            raise NotImplementedError(f"Unsupported HunyuanVideo transformer class: {name}")
        known = {f.name for f in fields(HunyuanVideoTransformerConfig)}
        values = {k: v for k, v in body.items() if k in known}
        if "rope_axes_dim" in values:
            values["rope_axes_dim"] = tuple(values["rope_axes_dim"])
        return cls(**{**values, **overrides})
