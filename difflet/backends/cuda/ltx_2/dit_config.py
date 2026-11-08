"""Backend-neutral LTX-2.0 transformer geometry and one inference shape.

The checkpoint's geometry is kept separate from accelerator topology. This config
supports the dense audiovisual LTX-2.0 architecture and the diffusers checkpoint
layout; it does not imply compatibility with other versions of the LTX family.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from difflet.models.ltx_2.contract import LTX_2_DEFAULT_TEXT_SEQ_LEN


@dataclass
class Ltx2DiTConfig:
    # Diffusers transformer geometry (transformer/config.json).
    in_channels: int = 128
    out_channels: int | None = 128
    patch_size: int = 1
    patch_size_t: int = 1
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    cross_attention_dim: int = 4096
    vae_scale_factors: tuple[int, int, int] = (8, 32, 32)
    pos_embed_max_pos: int = 20
    base_height: int = 2048
    base_width: int = 2048
    gated_attn: bool = False
    cross_attn_mod: bool = False
    audio_in_channels: int = 128
    audio_out_channels: int | None = 128
    audio_patch_size: int = 1
    audio_patch_size_t: int = 1
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_cross_attention_dim: int = 2048
    audio_scale_factor: int = 4
    audio_pos_embed_max_pos: int = 20
    audio_sampling_rate: int = 16000
    audio_hop_length: int = 160
    audio_gated_attn: bool = False
    audio_cross_attn_mod: bool = False
    num_layers: int = 48
    activation_fn: str = "gelu-approximate"
    qk_norm: str = "rms_norm_across_heads"
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-6
    caption_channels: int = 3840
    attention_bias: bool = True
    attention_out_bias: bool = True
    rope_theta: float = 10000.0
    rope_double_precision: bool = True
    causal_offset: int = 1
    timestep_scale_multiplier: int = 1000
    cross_attn_timestep_scale_multiplier: int = 1000
    rope_type: str = "split"
    use_prompt_embeddings: bool = True
    perturbed_attn: bool = False
    # One resident inference shape.
    height: int = 512
    width: int = 768
    num_frames: int = 121
    frame_rate: float = 24.0
    #: Text rows per prompt, as returned by the connectors.
    text_seq_len: int = LTX_2_DEFAULT_TEXT_SEQ_LEN
    #: The audio latent frame count; None derives it from the clip's duration.
    audio_num_frames: int | None = None
    audio_vae_temporal_compression_ratio: int = 4
    batch_size: int = 1
    #: LTX-2.0 semantics (False): the cross-modality modulation uses the stream's own timestep.
    use_cross_timestep: bool = False

    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        self.out_channels = self.in_channels if self.out_channels is None else self.out_channels
        self.audio_out_channels = (
            self.audio_in_channels if self.audio_out_channels is None else self.audio_out_channels
        )
        self.vae_scale_factors = tuple(self.vae_scale_factors)
        if not math.isfinite(self.frame_rate) or self.frame_rate <= 0:
            raise ValueError("LTX-2 frame_rate must be finite and positive")
        for name in (
            "audio_sampling_rate",
            "audio_hop_length",
            "audio_vae_temporal_compression_ratio",
        ):
            value = getattr(self, name)
            if (not isinstance(value, int) or isinstance(value, bool)) or value < 1:
                raise ValueError(f"LTX-2 {name} must be a positive integer")
        if (
            not isinstance(self.num_frames, int) or isinstance(self.num_frames, bool)
        ) or self.num_frames < 1:
            raise ValueError("LTX-2 num_frames must be a positive integer")
        if self.audio_num_frames is None:
            self.audio_num_frames = round(
                self.num_frames
                / self.frame_rate
                * self.audio_sampling_rate
                / self.audio_hop_length
                / self.audio_vae_temporal_compression_ratio
            )
        self.validate()

    @property
    def latent_num_frames(self) -> int:
        return (self.num_frames - 1) // self.vae_scale_factors[0] + 1

    @property
    def latent_height(self) -> int:
        return self.height // self.vae_scale_factors[1]

    @property
    def latent_width(self) -> int:
        return self.width // self.vae_scale_factors[2]

    @property
    def video_seq_len(self) -> int:
        return self.latent_num_frames * self.latent_height * self.latent_width

    @property
    def audio_seq_len(self) -> int:
        return self.audio_num_frames

    @property
    def video_text_dim(self) -> int:
        return self.caption_channels

    @property
    def audio_text_dim(self) -> int:
        return self.caption_channels

    def validate(self) -> None:
        for name in (
            "gated_attn",
            "audio_gated_attn",
            "cross_attn_mod",
            "audio_cross_attn_mod",
            "perturbed_attn",
            "use_cross_timestep",
        ):
            if getattr(self, name):
                raise NotImplementedError(f"LTX-2.0 GPU inference does not support {name}")
        if not self.use_prompt_embeddings:
            raise NotImplementedError("LTX-2.0 requires use_prompt_embeddings=True")
        if any(
            (isinstance(getattr(self, name), bool) or getattr(self, name) != 1)
            for name in (
                "patch_size",
                "patch_size_t",
                "audio_patch_size",
                "audio_patch_size_t",
            )
        ):
            raise NotImplementedError("LTX-2 GPU inference requires patch sizes of 1")
        if self.vae_scale_factors != (8, 32, 32):
            raise NotImplementedError("LTX-2 GPU inference requires VAE scale factors (8, 32, 32)")
        if isinstance(self.batch_size, bool) or self.batch_size != 1:
            raise NotImplementedError("LTX-2 GPU inference supports batch_size=1")
        if self.torch_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("LTX-2 transformer dtype must be bfloat16 or float32")
        if self.rope_type not in ("split", "interleaved"):
            raise ValueError(f"Unsupported LTX-2 rope_type={self.rope_type!r}")
        for name in ("height", "width"):
            value = getattr(self, name)
            if (not isinstance(value, int) or isinstance(value, bool)) or value <= 0 or value % 32:
                raise ValueError(f"LTX-2 {name} must be a positive multiple of 32")
        if (
            (not isinstance(self.num_frames, int) or isinstance(self.num_frames, bool))
            or self.num_frames < 1
            or (self.num_frames - 1) % 8
        ):
            raise ValueError("LTX-2 num_frames must be 8k+1")
        for name in (
            "num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "cross_attention_dim",
            "caption_channels",
            "text_seq_len",
            "audio_num_attention_heads",
            "audio_attention_head_dim",
            "audio_in_channels",
            "audio_out_channels",
            "audio_cross_attention_dim",
            "audio_num_frames",
            "audio_sampling_rate",
            "audio_hop_length",
            "audio_scale_factor",
            "audio_vae_temporal_compression_ratio",
        ):
            value = getattr(self, name)
            if (not isinstance(value, int) or isinstance(value, bool)) or value < 1:
                raise ValueError(f"LTX-2 {name} must be a positive integer")

    @classmethod
    def from_pretrained(cls, transformer_path, **shape_and_dtype) -> "Ltx2DiTConfig":
        raw = json.loads((Path(transformer_path) / "config.json").read_text())
        class_name = raw.get("_class_name", "LTX2VideoTransformer3DModel")
        if class_name != "LTX2VideoTransformer3DModel":
            raise ValueError(f"Unsupported LTX-2 transformer class {class_name!r}")
        names = {field.name for field in fields(cls)}
        unknown = {key for key in raw if key not in names and not key.startswith("_")}
        # Diffusers writes the serialized parameter dtype as ordinary metadata.
        unknown -= {"torch_dtype", "dtype"}
        if unknown:
            raise ValueError(f"Unsupported LTX-2 checkpoint configuration keys: {sorted(unknown)}")
        runtime_names = {
            "height",
            "width",
            "num_frames",
            "frame_rate",
            "text_seq_len",
            "audio_num_frames",
            "audio_vae_temporal_compression_ratio",
            "batch_size",
            "torch_dtype",
        }
        unknown_overrides = set(shape_and_dtype) - runtime_names
        if unknown_overrides:
            raise ValueError(f"Unsupported LTX-2 config overrides: {sorted(unknown_overrides)}")
        geometry = {
            key: value for key, value in raw.items() if key in names and key not in runtime_names
        }
        return cls(**geometry, **shape_and_dtype)


__all__ = ["Ltx2DiTConfig"]
