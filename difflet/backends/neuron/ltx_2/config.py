"""Shape and topology config for the LTX-2 DiT on the neuron backend.

A copy of the geometry ``LTX2TransformerInferenceConfig`` derives on Trainium
(``difflet/backends/trainium/ltx_2/transformer.py``), not an import: that class is an NxD
``InferenceConfig``, and the neuron backend must not import NxD. It holds the diffusers
hyperparameters of ``transformer/config.json``, the one static runtime shape (height, width,
frames, the zero-padded text length) and the latent arithmetic the two token streams are derived
from: ``video_seq_len`` packed video tokens (``(F - 1) / 8 + 1`` latent frames times
``H / 32 x W / 32``) and ``audio_seq_len`` audio latent frames (the clip's duration at 16 kHz,
hop 160, 4x temporal compression). The defaults are LTX-2 itself (48 blocks, 32 video heads x
128, 32 audio heads x 64) at the registry's 512x768x121, so ``NeuronLtx2Config()`` describes the
real model.

The backend-neutral ``_LTX2TransformerTraceModule`` (``difflet/models/ltx_2/modeling_ltx_2.py``)
duck-types its config: it reads the geometry attributes below by name, ``latent_num_frames``,
``latent_height``, ``latent_width``, ``frame_rate``, ``audio_num_frames`` and
``use_cross_timestep``. The topology flags are declared here so an unsupported mode fails in
``validate()`` with a message instead of building a model that ignores it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch

#: The diffusers LTX-2 pipeline's ``max_sequence_length`` and Difflet's text budget on every
#: backend (``LTX_2_DEFAULT_TEXT_SEQ_LEN``, ``difflet/models/ltx_2/application.py``); the
#: connectors return exactly this many rows (padding replaced by learned registers).
LTX_2_TEXT_SEQ_LEN = 1024


@dataclass
class NeuronLtx2Config:
    """LTX-2 geometry plus the neuron runtime's shape and topology."""

    # --- diffusers geometry (transformer/config.json) ---
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
    #: The checkpoint's own block count, set by ``from_pretrained`` before a ``num_layers``
    #: override: an on-device parity check that builds only the first N blocks of the real
    #: checkpoint skips the rest (reported as skipped by the application, not as unexpected).
    checkpoint_num_layers: int | None = None

    # --- difflet runtime shape (one static shape per process) ---
    height: int = 512
    width: int = 768
    num_frames: int = 121
    frame_rate: float = 24.0
    #: Text rows per prompt, as the connectors return them (the DiT attends them unmasked).
    text_seq_len: int = LTX_2_TEXT_SEQ_LEN
    #: The audio latent frame count; None derives it from the clip's duration.
    audio_num_frames: int | None = None
    audio_vae_temporal_compression_ratio: int = 4
    batch_size: int = 1
    #: LTX-2.0 semantics (False): the cross-modality modulation uses the stream's own timestep.
    use_cross_timestep: bool = False

    # --- topology ---
    tp_degree: int = 1
    dp_degree: int = 1
    cfg_parallel_enabled: bool = False
    context_parallel_enabled: bool = False
    cp_mode: str = "gather_kv"
    sp_enabled: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.out_channels is None:
            self.out_channels = self.in_channels
        if self.audio_out_channels is None:
            self.audio_out_channels = self.audio_in_channels
        self.vae_scale_factors = tuple(int(v) for v in self.vae_scale_factors)
        if self.audio_num_frames is None:
            self.audio_num_frames = self._infer_audio_num_frames()

    # --- derived geometry (mirrors LTX2TransformerInferenceConfig) ---
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
        # Audio latents are packed [B, audio_frames, channels * mel_bins]: the mel axis is part
        # of the feature dimension, not the sequence.
        return int(self.audio_num_frames) // int(self.audio_patch_size_t)

    @property
    def video_text_dim(self) -> int:
        return int(self.caption_channels if self.use_prompt_embeddings
                   else self.cross_attention_dim)

    @property
    def audio_text_dim(self) -> int:
        return int(self.caption_channels if self.use_prompt_embeddings
                   else self.audio_cross_attention_dim)

    @property
    def dit_batch(self) -> int:
        """The batch of one DiT call: the CFG-parallel ``[unconditional, conditional]`` pair,
        which the module splits over the CFG group, or ``batch_size``."""
        return 2 if self.cfg_parallel_enabled else int(self.batch_size)

    def _infer_audio_num_frames(self) -> int:
        duration_s = int(self.num_frames) / float(self.frame_rate)
        per_second = (int(self.audio_sampling_rate) / int(self.audio_hop_length)
                      / float(self.audio_vae_temporal_compression_ratio))
        return round(duration_s * per_second)

    def validate(self) -> None:
        if int(self.patch_size) != 1 or int(self.patch_size_t) != 1:
            raise NotImplementedError("LTX-2 supports video patch_size=patch_size_t=1")
        if int(self.audio_patch_size_t) != 1:
            raise NotImplementedError("LTX-2 supports audio_patch_size_t=1")
        if self.gated_attn or self.audio_gated_attn:
            raise NotImplementedError(
                "gated attention (LTX-2.3 checkpoints) is not supported on the neuron backend: "
                "its per-head gates are not sharded by the TP attention processor"
            )
        if self.perturbed_attn:
            raise NotImplementedError(
                "perturbed_attn (STG) is not supported on the neuron backend: the TP attention "
                "processor runs no perturbation, as on Trainium"
            )
        spatial = int(self.vae_scale_factors[1])
        for name in ("height", "width"):
            value = int(getattr(self, name))
            if value <= 0 or value % spatial:
                raise ValueError(
                    f"LTX-2 {name}={value} must be a positive multiple of {spatial} (the VAE's "
                    "spatial compression)"
                )
        temporal = int(self.vae_scale_factors[0])
        if int(self.num_frames) < 1 or (int(self.num_frames) - 1) % temporal:
            raise ValueError(
                f"LTX-2 num_frames={self.num_frames} must be {temporal}k+1 (the VAE's causal "
                "temporal compression keeps the first frame)"
            )
        if int(self.audio_num_frames) < 1:
            raise ValueError(f"LTX-2 audio_num_frames={self.audio_num_frames} must be >= 1")
        if int(self.num_layers) < 1:
            raise ValueError(f"LTX-2 num_layers={self.num_layers} must be >= 1")
        if self.checkpoint_num_layers is not None and int(self.num_layers) > int(
            self.checkpoint_num_layers
        ):
            raise ValueError(
                f"LTX-2 num_layers={self.num_layers} exceeds the checkpoint's "
                f"{self.checkpoint_num_layers} blocks"
            )
        tp = int(self.tp_degree)
        for what, heads in (("video", self.num_attention_heads),
                            ("audio", self.audio_num_attention_heads)):
            if int(heads) % tp:
                raise ValueError(
                    f"LTX-2 has {heads} {what} attention heads, which does not divide tp={tp}"
                )
        if int(self.dp_degree) > 1:
            raise NotImplementedError(
                "the neuron backend runs tensor and CFG parallelism within one LTX-2 "
                f"application; unsupported: dp_degree={self.dp_degree} (data parallelism runs "
                "as separate replicas)"
            )
        if self.context_parallel_enabled:
            raise NotImplementedError(
                "context parallelism is not supported for LTX-2 on the neuron backend"
            )
        if self.sp_enabled:
            raise NotImplementedError(
                "sequence parallelism is not supported for LTX-2 on the neuron backend"
            )

    @classmethod
    def from_pretrained(cls, transformer_path, **overrides) -> "NeuronLtx2Config":
        """Build from a diffusers ``transformer/config.json``; keys this class does not declare
        (``_class_name``, ...) are ignored. ``checkpoint_num_layers`` records the file's
        ``num_layers`` before the overrides, so ``num_layers=N`` builds the first N blocks of
        the same checkpoint."""
        raw = json.loads((Path(transformer_path) / "config.json").read_text())
        names = {f.name for f in fields(cls)}
        kept = {k: v for k, v in raw.items() if k in names}
        if "num_layers" in kept:
            kept["checkpoint_num_layers"] = int(kept["num_layers"])
        kept.update(overrides)
        config = cls(**kept)
        config.validate()
        return config


__all__ = ["LTX_2_TEXT_SEQ_LEN", "NeuronLtx2Config"]
