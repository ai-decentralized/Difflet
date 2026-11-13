"""The HunyuanVideo 1.5 text-to-video transformer config and its resident shape."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch


@dataclass
class HunyuanVideo15DiTConfig:
    in_channels: int = 65
    out_channels: int = 32
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    num_layers: int = 54
    num_refiner_layers: int = 2
    mlp_ratio: float = 4.0
    patch_size: int = 1
    patch_size_t: int = 1
    qk_norm: str = "rms_norm"
    text_embed_dim: int = 3584
    text_embed_2_dim: int = 1472
    image_embed_dim: int = 1152
    rope_theta: float = 256.0
    rope_axes_dim: tuple[int, ...] = (16, 56, 56)
    target_size: int = 640
    task_type: str = "t2v"
    use_meanflow: bool = False
    height: int = 480
    width: int = 848
    num_frames: int = 121
    text_seq_len: int = 1000
    text_2_seq_len: int = 256
    image_tokens: int = 729
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        # The one owner of the resident shape: the application builds this config before any
        # weights load.
        if min(self.height, self.width, self.num_frames) <= 0:
            raise ValueError("HunyuanVideo 1.5 height, width and num_frames must be positive")
        if self.height % 16 or self.width % 16:
            raise ValueError("HunyuanVideo 1.5 height and width must be multiples of 16")
        if (self.num_frames - 1) % 4:
            raise ValueError("HunyuanVideo 1.5 num_frames must be 4n+1")
        if self.task_type != "t2v" or self.use_meanflow:
            raise NotImplementedError("HunyuanVideo 1.5 supports standard text-to-video only")
        if self.patch_size != 1 or self.patch_size_t != 1 or self.qk_norm != "rms_norm":
            raise NotImplementedError(
                "HunyuanVideo 1.5 supports patch size 1 with RMSNorm Q/K normalization only"
            )
        self.rope_axes_dim = tuple(self.rope_axes_dim)

    @property
    def latent_height(self):
        return self.height // 16

    @property
    def latent_width(self):
        return self.width // 16

    @property
    def latent_frames(self):
        return (self.num_frames - 1) // 4 + 1

    @classmethod
    def from_pretrained(cls, transformer_dir, **overrides):
        body = json.loads((Path(transformer_dir) / "config.json").read_text())
        if (
            body.get("_class_name", "HunyuanVideo15Transformer3DModel")
            != "HunyuanVideo15Transformer3DModel"
        ):
            raise NotImplementedError("Unsupported HunyuanVideo 1.5 transformer class")
        known = {f.name for f in fields(cls)}
        return cls(**{**{k: v for k, v in body.items() if k in known}, **overrides})
