"""Shape and topology config for the HunyuanVideo 1.0 DiT on the neuron backend.

The diffusers hyperparameters of ``transformer/config.json``, the one static runtime shape
(height, width, frames, the right-padded text length) and the latent arithmetic the image tokens
are derived from: ``(F - 1) / 4 + 1`` latent frames times ``H / 8 x W / 8``, patched 1x2x2, so
``num_image_tokens`` is 16 x 20 x 32 = 10,240 at the registry's 320x512x61. The defaults are
HunyuanVideo itself (20 dual-stream and 40 single-stream blocks, 24 heads x 128), so
``NeuronHunyuanVideoConfig()`` describes the real model.

The backend-neutral ``HunyuanVideoTransformer3DModel``
(``difflet/models/hunyuan_video/modeling_hunyuan_video.py``) duck-types its config: it reads the
diffusers keys below by name and, with ``getattr``, ``context_parallel_enabled``, ``cp_mode``,
``sp_enabled`` and ``image_condition_type``; under CP it hands the whole config to
``init_parallel_mesh``, whose neuron ``mesh_spec_from_config`` reads ``tp_degree`` and
``cfg_parallel_enabled`` and takes cp as whatever the world leaves. The topology flags are
declared here so an unsupported or ragged mode fails in ``validate()`` with a message, before
the mesh and the meta build, instead of building a model that ignores it or fails in its first
forward. The rules shared with Wan, Qwen-Image and LTX-2 come from
``difflet/backends/neuron/core/config_checks.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from difflet.backends.neuron.core.config_checks import (
    check_cp_enabled,
    check_cp_mode,
    check_cp_split,
    check_even_split,
    check_heads_divide_tp,
    check_layer_count,
    check_ulysses_heads,
    exclusive_modes,
)


@dataclass
class NeuronHunyuanVideoConfig:
    """HunyuanVideo geometry plus the neuron runtime's shape and topology."""

    # --- diffusers geometry (transformer/config.json) ---
    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    num_layers: int = 20
    num_single_layers: int = 40
    num_refiner_layers: int = 2
    mlp_ratio: float = 4.0
    patch_size: int = 2
    patch_size_t: int = 1
    qk_norm: str = "rms_norm"
    guidance_embeds: bool = True
    text_embed_dim: int = 4096
    pooled_projection_dim: int = 768
    rope_theta: float = 256.0
    rope_axes_dim: tuple[int, ...] = (16, 56, 56)
    #: Set by the image-to-video checkpoints only; the text-to-video snapshot has none.
    image_condition_type: str | None = None
    #: The checkpoint's own block counts (``from_pretrained`` records the file's). A
    #: ``num_layers`` or ``num_single_layers`` below them builds the first blocks of each stack
    #: and ``NeuronHunyuanVideoTransformerApplication`` reports the rest under
    #: ``load_report["skipped"]`` rather than as unexpected.
    checkpoint_num_layers: int | None = 20
    checkpoint_num_single_layers: int | None = 40

    # --- difflet runtime shape (one static shape per process) ---
    height: int = 320
    width: int = 512
    num_frames: int = 61
    batch_size: int = 1
    #: Llama tokens per prompt (diffusers' ``max_sequence_length``), right-padded; the
    #: ``encoder_attention_mask`` marks the valid ones.
    text_seq_len: int = 256
    # HunyuanVideo's VAE downsamples 8x spatially and 4x temporally (causal: the first frame
    # is kept).
    vae_scale_factor_spatial: int = 8
    vae_scale_factor_temporal: int = 4

    # --- topology ---
    tp_degree: int = 1
    #: The CP degree when known (the application fills it from the parallel config); None
    #: leaves the cp-dependent checks to the mesh, which derives cp from the world size.
    cp_degree: int | None = None
    context_parallel_enabled: bool = False
    cp_mode: str = "gather_kv"
    sp_enabled: bool = False
    cfg_parallel_enabled: bool = False
    # Keep Neuron's existing bound-based attention when the shared port reads its policy.
    joint_attention_mask_mode: str = "bounds"

    def __post_init__(self) -> None:
        self.rope_axes_dim = tuple(int(d) for d in self.rope_axes_dim)

    # --- derived geometry ---
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
    def num_image_tokens(self) -> int:
        p, p_t = int(self.patch_size), int(self.patch_size_t)
        return ((self.latent_frames // p_t) * (self.latent_height // p)
                * (self.latent_width // p))

    @property
    def inner_dim(self) -> int:
        return int(self.num_attention_heads) * int(self.attention_head_dim)

    def _token_layout(self) -> str:
        p, p_t = int(self.patch_size), int(self.patch_size_t)
        return (f"{self.latent_frames // p_t}x{self.latent_height // p}x"
                f"{self.latent_width // p} patched latent")

    def validate(self) -> None:
        check_layer_count("HunyuanVideo", self.num_layers, self.checkpoint_num_layers)
        check_layer_count("HunyuanVideo single-stream", self.num_single_layers,
                          self.checkpoint_num_single_layers)
        if self.image_condition_type is not None:
            raise NotImplementedError(
                f"image_condition_type={self.image_condition_type!r} (an image-to-video "
                "checkpoint) is not supported on the neuron backend; HunyuanVideo runs "
                "text-to-video only"
            )
        if self.cfg_parallel_enabled:
            raise NotImplementedError(
                "HunyuanVideo is guidance-distilled (one forward pass with the guidance scale "
                "baked into the timestep embedding); CFG parallelism needs true two-pass "
                "classifier-free guidance and does not apply"
            )
        # Check the *requested* pixel size: the latent dims are floor divisions, so an
        # unaligned height would silently generate a smaller video than asked for.
        multiple = int(self.vae_scale_factor_spatial) * int(self.patch_size)
        for name in ("height", "width"):
            value = int(getattr(self, name))
            if value <= 0 or value % multiple:
                raise ValueError(
                    f"HunyuanVideo {name}={value} must be a positive multiple of {multiple} "
                    f"(vae {self.vae_scale_factor_spatial}x, patch {self.patch_size})"
                )
        temporal = int(self.vae_scale_factor_temporal)
        if int(self.num_frames) < 1 or (int(self.num_frames) - 1) % temporal:
            raise ValueError(
                f"HunyuanVideo num_frames={self.num_frames} must be {temporal}k+1 (the VAE's "
                "causal temporal compression keeps the first frame)"
            )
        if self.latent_frames % int(self.patch_size_t):
            raise ValueError(
                f"HunyuanVideo latent frames {self.latent_frames} is not divisible by the "
                f"temporal patch {self.patch_size_t}"
            )
        check_heads_divide_tp("HunyuanVideo", self.num_attention_heads, self.tp_degree)
        self._validate_parallel()

    def _validate_parallel(self) -> None:
        """The parallel modes' divisibility rules, mirroring ``NeuronWanConfig``.

        The port splits the image tokens with ``scatter_to_process_group_spmd``, which has no
        padding: CP over the CP group, SP over the TP group (SP shards only the image stream of
        the dual-stream blocks; the text stays whole). A ragged split would surface as a shape
        error in the first forward's collective. Ulysses then shards this rank's heads over CP.
        SP and CP both shard the image sequence, which the port itself refuses in ``__init__``;
        it is checked here so the error comes before the mesh and the meta build.
        """
        check_cp_mode(self.cp_mode)
        if self.context_parallel_enabled and self.sp_enabled:
            raise exclusive_modes("HunyuanVideo", "sp_enabled", "context_parallel_enabled",
                                  "both shard the image sequence")
        heads, tp = int(self.num_attention_heads), int(self.tp_degree)
        tokens = self.num_image_tokens
        if self.sp_enabled:
            check_even_split(
                f"HunyuanVideo sequence parallelism splits num_image_tokens={tokens} "
                f"({self._token_layout()})", tokens, "tp", tp)
        cp = None if self.cp_degree is None else int(self.cp_degree)
        check_cp_enabled(cp, self.context_parallel_enabled)
        if not self.context_parallel_enabled or cp is None or cp <= 1:
            return
        check_cp_split("HunyuanVideo", "num_image_tokens", tokens, self._token_layout(), cp)
        if self.cp_mode == "ulysses":
            check_ulysses_heads(heads, tp, cp)

    @classmethod
    def from_pretrained(cls, transformer_path, **overrides) -> "NeuronHunyuanVideoConfig":
        """Build from a diffusers ``transformer/config.json``; keys this class does not declare
        (``_class_name``, ...) are ignored. ``checkpoint_num_layers`` and
        ``checkpoint_num_single_layers`` record the file's block counts before the overrides, so
        ``num_layers=N`` builds the first N blocks of the same checkpoint."""
        raw = json.loads((Path(transformer_path) / "config.json").read_text())
        names = {f.name for f in fields(cls)}
        kept = {k: v for k, v in raw.items() if k in names}
        for stack in ("num_layers", "num_single_layers"):
            if stack in kept:
                kept[f"checkpoint_{stack}"] = int(kept[stack])
        kept.update(overrides)
        config = cls(**kept)
        config.validate()
        return config


__all__ = ["NeuronHunyuanVideoConfig"]
