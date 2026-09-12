"""Geometry config for the LTX-2 audiovisual DiT on TPU.

LTX-2's DiT is diffusers' own ``LTX2VideoTransformer3DModel`` (no difflet
fork), so unlike Qwen/Wan/HunyuanVideo there is no modeling dataclass to
extend: this mirrors the Trainium ``LTX2TransformerInferenceConfig`` — the
transformer's ``config.json`` fields as attributes plus the runtime shape,
text lengths, audio geometry, tp degree and dtype, with the same derived
properties — as a plain object ``build_ltx2_transformer`` and
``validate_ltx_2_dit_inputs`` can read.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from difflet.models.ltx_2.application import LTX_2_DEFAULT_TEXT_SEQ_LEN


class TpuLTX2Config:
    """``transformer/config.json`` + runtime shape/topology for the TPU lifecycle."""

    def __init__(
        self,
        raw: dict[str, Any],
        *,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        batch_size: int = 1,
        text_seq_len: int = LTX_2_DEFAULT_TEXT_SEQ_LEN,
        audio_text_seq_len: int | None = None,
        audio_num_frames: int | None = None,
        frame_rate: float = 24.0,
        tp_degree: int = 1,
        torch_dtype: torch.dtype = torch.bfloat16,
        context_parallel_enabled: bool = False,
        cfg_parallel_enabled: bool = False,
        sp_enabled: bool = False,
    ) -> None:
        for key, value in raw.items():
            if not key.startswith("_"):
                setattr(self, key, value)
        if getattr(self, "out_channels", None) is None:
            self.out_channels = self.in_channels
        if getattr(self, "audio_out_channels", None) is None:
            self.audio_out_channels = self.audio_in_channels
        if isinstance(getattr(self, "vae_scale_factors", None), list):
            self.vae_scale_factors = tuple(self.vae_scale_factors)
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.batch_size = int(batch_size)
        self.text_seq_len = int(text_seq_len)
        self.audio_text_seq_len = int(audio_text_seq_len or text_seq_len)
        self.frame_rate = float(frame_rate)
        self.tp_degree = int(tp_degree)
        self.torch_dtype = torch_dtype
        self.context_parallel_enabled = bool(context_parallel_enabled)
        self.cfg_parallel_enabled = bool(cfg_parallel_enabled)
        self.sp_enabled = bool(sp_enabled)
        self.audio_num_frames = (
            int(audio_num_frames) if audio_num_frames is not None else self._infer_audio_num_frames()
        )
        use_prompt = bool(getattr(self, "use_prompt_embeddings", True))
        self.video_text_dim = int(self.caption_channels if use_prompt else self.cross_attention_dim)
        self.audio_text_dim = int(
            self.caption_channels if use_prompt else self.audio_cross_attention_dim
        )
        # validate_ltx_2_dit_inputs / dit_input_contract read the batch size
        # through the Neuron-shaped ``config.neuron_config``.
        self.neuron_config = SimpleNamespace(
            batch_size=self.batch_size, torch_dtype=torch_dtype, tp_degree=self.tp_degree
        )

    # --- derived geometry (same definitions as the Trainium config) ---
    @property
    def latent_num_frames(self) -> int:
        return (int(self.num_frames) - 1) // int(self.vae_scale_factors[0]) + 1

    @property
    def latent_height(self) -> int:
        return int(self.height) // int(self.vae_scale_factors[1])

    @property
    def latent_width(self) -> int:
        return int(self.width) // int(self.vae_scale_factors[2])

    @property
    def video_seq_len(self) -> int:
        return (
            (self.latent_num_frames // int(self.patch_size_t))
            * (self.latent_height // int(self.patch_size))
            * (self.latent_width // int(self.patch_size))
        )

    @property
    def audio_seq_len(self) -> int:
        return int(self.audio_num_frames) // int(self.audio_patch_size_t)

    @property
    def inner_dim(self) -> int:
        return int(self.num_attention_heads) * int(self.attention_head_dim)

    def _infer_audio_num_frames(self) -> int:
        duration_s = int(self.num_frames) / float(self.frame_rate)
        per_second = (
            int(self.audio_sampling_rate)
            / int(self.audio_hop_length)
            / float(getattr(self, "audio_vae_temporal_compression_ratio", 4))
        )
        return round(duration_s * per_second)

    def validate(self) -> None:
        tp = int(self.tp_degree)
        for name in ("num_attention_heads", "audio_num_attention_heads"):
            if int(getattr(self, name)) % tp:
                raise ValueError(f"LTX-2 {name}={getattr(self, name)} does not divide tp={tp}")
        if int(self.patch_size) != 1 or int(self.patch_size_t) != 1:
            raise NotImplementedError("LTX-2 supports video patch_size=patch_size_t=1")
        if int(self.audio_patch_size_t) != 1:
            raise NotImplementedError("LTX-2 supports audio_patch_size_t=1")
        if int(self.height) % int(self.vae_scale_factors[1]):
            raise ValueError("LTX-2 height must be divisible by the VAE spatial scale")
        if int(self.width) % int(self.vae_scale_factors[2]):
            raise ValueError("LTX-2 width must be divisible by the VAE spatial scale")
        if (int(self.num_frames) - 1) % int(self.vae_scale_factors[0]):
            raise ValueError(
                f"LTX-2 num_frames={self.num_frames} must satisfy "
                f"(num_frames - 1) % {self.vae_scale_factors[0]} == 0"
            )
        if bool(getattr(self, "gated_attn", False)) or bool(getattr(self, "perturbed_attn", False)):
            raise NotImplementedError("LTX-2 TP sharding does not support gated/perturbed attention")
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
    def from_pretrained(cls, transformer_path, **overrides) -> "TpuLTX2Config":
        raw = json.loads((Path(transformer_path) / "config.json").read_text())
        config = cls(raw, **overrides)
        config.validate()
        return config


__all__ = ["TpuLTX2Config"]
