"""LTX-2.0 transformer geometry and the one resident inference shape of the CUDA component.

The architecture fields mirror diffusers' ``transformer/config.json`` for the dense audiovisual
LTX-2.0 checkpoint; other members of the LTX family (gated or cross-modulated attention) are
refused here. This config owns the shape constraints of the CUDA path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from difflet.models.ltx_2.defaults import (
    LTX_2_DEFAULT_HEIGHT,
    LTX_2_DEFAULT_NUM_FRAMES,
    LTX_2_DEFAULT_TEXT_SEQ_LEN,
    LTX_2_DEFAULT_WIDTH,
)

#: Fields set by the request, never by the checkpoint's config.json.
_RUNTIME_FIELDS = frozenset(
    {
        "height",
        "width",
        "num_frames",
        "frame_rate",
        "text_seq_len",
        "audio_num_frames",
        "audio_vae_temporal_compression_ratio",
        "torch_dtype",
    }
)

#: LTX-2.x features the 2.0 checkpoint does not use and this component does not implement.
_UNSUPPORTED_FEATURES = (
    "gated_attn",
    "audio_gated_attn",
    "cross_attn_mod",
    "audio_cross_attn_mod",
    "perturbed_attn",
    "use_cross_timestep",
)


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
    height: int = LTX_2_DEFAULT_HEIGHT
    width: int = LTX_2_DEFAULT_WIDTH
    num_frames: int = LTX_2_DEFAULT_NUM_FRAMES
    frame_rate: float = 24.0
    #: Text rows per prompt, as returned by the connectors.
    text_seq_len: int = LTX_2_DEFAULT_TEXT_SEQ_LEN
    #: The audio latent frame count; None derives it from the clip's duration.
    audio_num_frames: int | None = None
    audio_vae_temporal_compression_ratio: int = 4
    #: LTX-2.0 semantics (False): the cross-modality modulation uses the stream's own timestep.
    use_cross_timestep: bool = False

    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.out_channels is None:
            self.out_channels = self.in_channels
        if self.audio_out_channels is None:
            self.audio_out_channels = self.audio_in_channels
        self.vae_scale_factors = tuple(self.vae_scale_factors)
        for name in _UNSUPPORTED_FEATURES:
            if getattr(self, name):
                raise NotImplementedError(f"LTX-2.0 GPU inference does not support {name}")
        if not self.use_prompt_embeddings:
            raise NotImplementedError("LTX-2.0 requires use_prompt_embeddings=True")
        # The latent geometry below and the example inputs assume unpatched tokens.
        patches = (
            self.patch_size,
            self.patch_size_t,
            self.audio_patch_size,
            self.audio_patch_size_t,
        )
        if patches != (1, 1, 1, 1):
            raise NotImplementedError("LTX-2 GPU inference requires patch sizes of 1")
        if self.vae_scale_factors != (8, 32, 32):
            raise NotImplementedError("LTX-2 GPU inference requires VAE scale factors (8, 32, 32)")
        if self.height <= 0 or self.width <= 0 or self.height % 32 or self.width % 32:
            raise ValueError("LTX-2 height and width must be positive multiples of 32")
        if self.num_frames < 1 or (self.num_frames - 1) % 8:
            raise ValueError("LTX-2 num_frames must be 8k+1")
        if self.audio_num_frames is None:
            self.audio_num_frames = round(
                self.num_frames
                / self.frame_rate
                * self.audio_sampling_rate
                / self.audio_hop_length
                / self.audio_vae_temporal_compression_ratio
            )

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

    @classmethod
    def from_pretrained(cls, transformer_path, **runtime) -> "Ltx2DiTConfig":
        """The checkpoint's architecture plus the request's ``runtime`` shape and dtype.

        Keys this config does not model (diffusers metadata, fields a newer diffusers adds)
        are ignored; strict weight loading still refuses a checkpoint whose layers differ.
        """
        raw = json.loads((Path(transformer_path) / "config.json").read_text())
        class_name = raw.get("_class_name", "LTX2VideoTransformer3DModel")
        if class_name != "LTX2VideoTransformer3DModel":
            raise ValueError(f"Unsupported LTX-2 transformer class {class_name!r}")
        architecture = {f.name for f in fields(cls)} - _RUNTIME_FIELDS
        return cls(**{k: v for k, v in raw.items() if k in architecture}, **runtime)


__all__ = ["Ltx2DiTConfig"]
