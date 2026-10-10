"""Shape and topology config for the Wan DiT on the neuron backend.

A copy of ``TpuWanConfig`` (``difflet/backends/tpu/wan/config.py:29-133``), not an import, so
the two backends can diverge without editing the TPU tree: ``WanTransformerConfig`` (the
hardware-neutral hyperparameters, ``difflet/models/wan/modeling_wan.py``) plus the one static
runtime shape, the tp degree and the dtype, and the latent geometry the warm-up inputs are
built from. Only ``validate()``'s message for the parallel modes differs.

The parallel-mode flags are the ones ``WanTransformer3DModel`` reads with ``getattr``; they
are declared so that an unsupported mode fails here, with a message, instead of building a
model that ignores it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from difflet.models.wan.modeling_wan import WanTransformerConfig


@dataclass
class NeuronWanConfig(WanTransformerConfig):
    """``WanTransformerConfig`` plus the neuron runtime's shape and topology."""

    # --- difflet runtime shape (one static shape per process) ---
    height: int = 480
    width: int = 832
    num_frames: int = 9
    batch_size: int = 1
    text_seq_len: int = 512
    # Wan's VAE downsamples 8x spatially and 4x temporally (temperal_downsample
    # [False, True, True] in the checkpoint's vae/config.json).
    vae_scale_factor_spatial: int = 8
    vae_scale_factor_temporal: int = 4

    # --- topology ---
    tp_degree: int = 1
    torch_dtype: torch.dtype = torch.bfloat16
    context_parallel_enabled: bool = False
    cfg_parallel_enabled: bool = False
    sp_enabled: bool = False
    cp_mode: str = "gather_kv"

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
        p_t, p_h, p_w = self.patch_size
        return (
            (self.latent_frames // int(p_t))
            * (self.latent_height // int(p_h))
            * (self.latent_width // int(p_w))
        )

    def validate(self) -> None:
        # tp shards the attention heads, so it has to divide them evenly.
        if int(self.num_attention_heads) % int(self.tp_degree):
            raise ValueError(
                f"Wan has {self.num_attention_heads} attention heads, which "
                f"does not divide tp={self.tp_degree}"
            )
        # Both are sharded column-wise by the TP linears; a ragged split would
        # only surface as a shape mismatch deep inside the weight load.
        if int(self.ffn_dim) % int(self.tp_degree):
            raise ValueError(
                f"Wan ffn_dim={self.ffn_dim} does not divide tp={self.tp_degree}"
            )
        p_t, p_h, p_w = (int(v) for v in self.patch_size)
        # Check the *requested* pixel size, not just the latent grid: the
        # latent dims are floor divisions, so an unaligned height silently
        # rounds down and generates a different shape than the caller asked
        # for rather than failing.
        spatial = int(self.vae_scale_factor_spatial)
        for name, patch in (("height", p_h), ("width", p_w)):
            value = int(getattr(self, name))
            if value % (spatial * patch):
                raise ValueError(
                    f"Wan compile {name}={value} must be divisible by "
                    f"{spatial * patch} (vae {spatial}x, patch {patch})"
                )
        temporal = int(self.vae_scale_factor_temporal)
        if (int(self.num_frames) - 1) % temporal:
            raise ValueError(
                f"Wan num_frames={self.num_frames} must satisfy "
                f"(num_frames - 1) % {temporal} == 0"
            )
        if self.latent_frames % p_t:
            raise ValueError(
                f"Wan latent frames {self.latent_frames} is not divisible by "
                f"the temporal patch {p_t}"
            )
        for mode, enabled in (
            ("context parallelism", self.context_parallel_enabled),
            ("cfg parallelism", self.cfg_parallel_enabled),
            ("sequence parallelism", self.sp_enabled),
        ):
            if enabled:
                raise NotImplementedError(
                    "the neuron backend supports tensor parallelism only (phase 1); "
                    f"{mode} is not implemented"
                )

    @classmethod
    def from_pretrained(cls, transformer_path, **overrides) -> "NeuronWanConfig":
        """Build from a diffusers ``transformer/config.json``."""
        raw = json.loads((Path(transformer_path) / "config.json").read_text())
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kept = {k: v for k, v in raw.items() if k in fields}
        if isinstance(kept.get("patch_size"), list):
            kept["patch_size"] = tuple(kept["patch_size"])
        kept.update(overrides)
        config = cls(**kept)
        config.validate()
        return config


__all__ = ["NeuronWanConfig"]
