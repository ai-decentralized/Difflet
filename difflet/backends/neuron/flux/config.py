"""Shape and topology config for the Flux DiT on the neuron backend.

The backend-neutral ``NeuronFluxTransformer2DModel`` (``difflet/models/flux/modeling_flux.py``)
duck-types its config: it reads the diffusers geometry below, ``neuron_config.torch_dtype``, and the
parallel flags (``context_parallel_enabled``, ``cp_mode``, ``cfg_parallel_enabled``,
``sp_enabled``); with CP or CFG on it hands the whole config to ``init_parallel_mesh``, whose
neuron ``mesh_spec_from_config`` reads ``tp_degree``, ``cfg_parallel_enabled`` and ``dp_degree``.
The defaults are FLUX.1-dev (19 dual-stream + 38 single-stream blocks, 24 heads x 128) at
1024x1024 with 512 T5 tokens, so ``NeuronFluxConfig()`` describes the real model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import torch

from difflet.pipeline.parallel_config import CP_MODES


@dataclass
class NeuronFluxConfig:
    """Flux geometry plus the neuron runtime's shape and topology."""

    # --- diffusers geometry (transformer/config.json) ---
    patch_size: int = 1
    in_channels: int = 64
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = True
    axes_dims_rope: tuple[int, ...] = (16, 56, 56)
    # --- difflet runtime shape (one static shape per process) ---
    height: int = 1024
    width: int = 1024
    #: T5 tokens per prompt (FLUX.1-dev: max_sequence_length=512).
    text_seq_len: int = 512
    batch_size: int = 1
    vae_scale_factor: int = 8
    # --- topology ---
    tp_degree: int = 1
    cp_degree: int | None = None
    dp_degree: int = 1
    cfg_parallel_enabled: bool = False
    context_parallel_enabled: bool = False
    cp_mode: str = "gather_kv"
    sp_enabled: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    @property
    def neuron_config(self) -> SimpleNamespace:
        """What the model reads from an NxD ``neuron_config``: the dtype it builds in."""
        return SimpleNamespace(torch_dtype=self.torch_dtype)

    @property
    def inner_dim(self) -> int:
        return int(self.num_attention_heads) * int(self.attention_head_dim)

    @property
    def mlp_hidden_dim(self) -> int:
        return 4 * self.inner_dim  # the single-stream blocks' mlp_ratio=4.0

    @property
    def packed_height(self) -> int:
        # latents are (height / vae_scale_factor) and packed 2x2 into channels
        return int(self.height) // (int(self.vae_scale_factor) * 2)

    @property
    def packed_width(self) -> int:
        return int(self.width) // (int(self.vae_scale_factor) * 2)

    @property
    def image_seq_len(self) -> int:
        return self.packed_height * self.packed_width

    def validate(self) -> None:
        if sum(int(d) for d in self.axes_dims_rope) != int(self.attention_head_dim):
            raise ValueError(f"Flux axes_dims_rope {tuple(self.axes_dims_rope)} must sum to "
                             f"attention_head_dim={self.attention_head_dim}")
        if int(self.patch_size) != 1:
            raise NotImplementedError("Flux packs latents 2x2 outside the DiT; patch_size must be 1")
        multiple = int(self.vae_scale_factor) * 2
        for name in ("height", "width"):
            value = int(getattr(self, name))
            if value <= 0 or value % multiple:
                raise ValueError(f"Flux {name}={value} must be a positive multiple of {multiple}")
        heads, tp = int(self.num_attention_heads), int(self.tp_degree)
        if heads % tp:
            raise ValueError(f"Flux has {heads} attention heads, which does not divide tp={tp}")
        if int(self.dp_degree) > 1:
            raise NotImplementedError(f"dp_degree={self.dp_degree}: data parallelism runs "
                                      "replicas of a dp_degree=1 config")
        if self.cfg_parallel_enabled:
            raise NotImplementedError(
                "FLUX.1-dev is guidance-distilled (one forward per step, guidance as an input); "
                "CFG parallelism needs two-pass classifier-free guidance and does not apply")
        if self.sp_enabled and self.context_parallel_enabled:
            raise ValueError("sp_enabled and context_parallel_enabled are mutually exclusive")
        if self.cp_mode not in CP_MODES:
            raise ValueError(f"cp_mode must be one of {CP_MODES}, got {self.cp_mode!r}")
        cp = None if self.cp_degree is None else int(self.cp_degree)
        if self.context_parallel_enabled and cp is not None and cp > 1:
            if self.image_seq_len % cp:
                raise ValueError(f"Flux image_seq_len={self.image_seq_len} is not divisible by "
                                 f"cp_degree={cp}")
            if self.cp_mode == "ulysses" and (heads // tp) % cp:
                raise ValueError(f"cp_mode='ulysses' needs num_attention_heads={heads} divisible "
                                 f"by tp_degree * cp_degree = {tp * cp}")

    @classmethod
    def from_pretrained(cls, transformer_path, **overrides) -> "NeuronFluxConfig":
        """Build from a diffusers ``transformer/config.json``; undeclared keys are ignored."""
        body = json.loads((Path(transformer_path) / "config.json").read_text())
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in body.items() if k in names}
        if "axes_dims_rope" in values:
            values["axes_dims_rope"] = tuple(values["axes_dims_rope"])
        values.update(overrides)
        return cls(**values)


__all__ = ["NeuronFluxConfig"]
