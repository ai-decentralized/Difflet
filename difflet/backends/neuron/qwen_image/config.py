"""Shape and topology config for the Qwen-Image DiT on the neuron backend.

A copy of ``TpuQwenImageConfig``'s geometry (``difflet/backends/tpu/qwen_image/config.py``), not
an import, so the two backends can diverge without editing the TPU tree: the diffusers
hyperparameters, the one static runtime shape, and the patch/VAE arithmetic the packed sequence
is derived from. The defaults are Qwen-Image itself (60 blocks, 24 heads x 128) at 1024x1024, so
``NeuronQwenImageConfig()`` describes the real model.

The backend-neutral ``_QwenImageTransformerTraceModule``
(``difflet/models/qwen_image/modeling_qwen_image.py``) duck-types its config: it reads
``context_parallel_enabled``, ``cp_mode`` and ``sp_enabled`` to pick its attention path, and,
with CP on, hands the whole config to ``init_parallel_mesh``, whose neuron
``mesh_spec_from_config`` reads ``tp_degree``, ``cfg_parallel_enabled`` and ``dp_degree`` and
takes cp as whatever the world leaves (``parallel_mesh.py``). Those flags are declared here so
that an unsupported mode fails in ``validate()``, with a message, instead of building a model
that ignores it or a mesh that disagrees with the one the application base already built.
``compile_shapes`` and ``num_frames`` are what ``resolve_compile_shapes`` reads to precompute
the static RoPE table: None and None mean the one (height, width) image shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch

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
class NeuronQwenImageConfig:
    """Qwen-Image geometry plus the neuron runtime's shape and topology."""

    # --- diffusers geometry (transformer/config.json) ---
    patch_size: int = 2
    in_channels: int = 64
    out_channels: int = 16
    num_layers: int = 60
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 3584
    guidance_embeds: bool = False
    axes_dims_rope: tuple[int, ...] = (16, 56, 56)
    zero_cond_t: bool = False
    use_layer3d_rope: bool = False
    #: The checkpoint's own block count, set by ``from_pretrained`` before a ``num_layers``
    #: override: an on-device parity check that builds only the first N blocks of the real
    #: checkpoint skips the rest (reported as skipped by the application, not as unexpected).
    checkpoint_num_layers: int | None = None

    # --- difflet runtime shape (one static shape per process) ---
    height: int = 1024
    width: int = 1024
    #: Text tokens per prompt after zero-padding (``TEXT_SEQ_LEN``,
    #: ``difflet/common/orchestrators/qwen_image.py``). The DiT attends them unmasked.
    text_seq_len: int = 1024
    batch_size: int = 1
    vae_scale_factor: int = 8
    compile_shapes: tuple | None = None
    num_frames: int | None = None

    # --- topology ---
    tp_degree: int = 1
    #: The CP degree when known (the application sets it from the parallel config); None
    #: leaves the cp-dependent checks to the mesh, which derives cp from the world size.
    cp_degree: int | None = None
    dp_degree: int = 1
    cfg_parallel_enabled: bool = False
    context_parallel_enabled: bool = False
    cp_mode: str = "gather_kv"
    sp_enabled: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # --- derived geometry (mirrors TpuQwenImageConfig) ---
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

    @property
    def dit_batch(self) -> int:
        """The batch of one DiT call: the CFG-parallel ``[negative, conditional]`` pair, which
        the module splits over the CFG group, or ``batch_size``."""
        return 2 if self.cfg_parallel_enabled else int(self.batch_size)

    def validate(self) -> None:
        if sum(int(d) for d in self.axes_dims_rope) != int(self.attention_head_dim):
            raise ValueError(
                f"Qwen-Image axes_dims_rope {tuple(self.axes_dims_rope)} must sum to "
                f"attention_head_dim={self.attention_head_dim}"
            )
        if any(int(d) % 2 for d in self.axes_dims_rope):
            raise ValueError("Qwen-Image axes_dims_rope entries must be even")
        if int(self.patch_size) != 2:
            raise NotImplementedError("Qwen-Image currently supports patch_size=2")
        if self.guidance_embeds:
            # The warm-up and the per-step call carry no guidance tensor (Qwen-Image's
            # checkpoint has no guidance embedder), and diffusers' guidance embedder cannot run
            # without one.
            raise NotImplementedError(
                "guidance_embeds=True (a guidance-embedding checkpoint) is not supported on the "
                "neuron backend; Qwen-Image's transformer has guidance_embeds=false"
            )
        # Check the *requested* pixel size: the latent dims are floor divisions, so an
        # unaligned height would silently generate a smaller image than asked for.
        multiple = int(self.vae_scale_factor) * 2
        for name in ("height", "width"):
            value = int(getattr(self, name))
            if value <= 0 or value % multiple:
                raise ValueError(
                    f"Qwen-Image compile {name}={value} must be a positive multiple of "
                    f"{multiple} (vae {self.vae_scale_factor}x, patch {self.patch_size})"
                )
        check_layer_count("Qwen-Image", self.num_layers, self.checkpoint_num_layers)
        heads, tp = int(self.num_attention_heads), int(self.tp_degree)
        check_heads_divide_tp("Qwen-Image", heads, tp)
        if int(self.dp_degree) > 1:
            # DP is a set of separate replicas (torchrun launches), never an axis of one module.
            raise NotImplementedError(
                "the neuron backend supports tensor, sequence, context and CFG parallelism "
                f"within one application; unsupported: dp_degree={self.dp_degree} (data "
                "parallelism runs as separate replicas)"
            )
        self._validate_sequence_parallel()
        if self.cfg_parallel_enabled and self.context_parallel_enabled:
            # As for Wan (NeuronWanConfig) and the CLI (--cfg-parallel with --cp-degree > 1).
            # CFG parallelism serves opt-in true CFG only: the application refuses a call that
            # does not run it.
            raise exclusive_modes("Qwen-Image", "cfg_parallel_enabled",
                                  "context_parallel_enabled", "both consume the data-parallel lanes")
        check_cp_mode(self.cp_mode)
        cp = None if self.cp_degree is None else int(self.cp_degree)
        check_cp_enabled(cp, self.context_parallel_enabled)
        if self.context_parallel_enabled and cp is not None and cp > 1:
            # The trace module scatters the packed image tokens (and their RoPE rows) evenly
            # over the CP group; text stays replicated, so only the image length matters.
            check_cp_split("Qwen-Image", "image_seq_len", self.image_seq_len,
                           f"{self.packed_height}x{self.packed_width} packed", cp)
            if self.cp_mode == "ulysses":
                check_ulysses_heads(heads, tp, cp)

    def _validate_sequence_parallel(self) -> None:
        """Megatron SP's rules (``QwenImageSPTransformer2DModel``, the fork the trace module
        builds under ``sp_enabled``).

        The fork shards both residual streams over the TP group between the blocks: the packed
        image tokens and the zero-padded text rows, each along its own sequence axis, with
        ``scatter_to_process_group_spmd`` (no padding) and a ``reduce_scatter`` per TP region.
        So both lengths must divide tp, or the first forward fails in a collective. SP and CP
        are mutually exclusive, as for Wan (``NeuronWanConfig``): both shard the image sequence,
        and the fork's entry scatter assumes the full sequence on every rank of the TP group,
        which the CP scatter has already cut. ``zero_cond_t`` gives every image token its own
        modulation index, which the fork cannot shard; the released checkpoints have it off.
        At tp == 1 the fork is the dense model (its SP insertions are skipped), so nothing is
        checked there.
        """
        if not self.sp_enabled:
            return
        if self.context_parallel_enabled:
            raise exclusive_modes("Qwen-Image", "sp_enabled", "context_parallel_enabled",
                                  "both shard the image sequence; as for Wan")
        if self.zero_cond_t:
            raise NotImplementedError(
                "Qwen-Image sequence parallelism does not support zero_cond_t (a per-token "
                "modulation index the sequence shards cannot carry)"
            )
        tp = int(self.tp_degree)
        if tp <= 1:
            return
        check_even_split(
            f"Qwen-Image sequence parallelism splits image_seq_len={self.image_seq_len} "
            f"({self.packed_height}x{self.packed_width} packed)", self.image_seq_len, "tp", tp)
        check_even_split(
            "Qwen-Image sequence parallelism splits the zero-padded text_seq_len="
            f"{self.text_seq_len}", int(self.text_seq_len), "tp", tp)

    @classmethod
    def from_pretrained(cls, transformer_path, **overrides) -> "NeuronQwenImageConfig":
        """Build from a diffusers ``transformer/config.json``; keys this class does not declare
        (``pooled_projection_dim``, ``_class_name``, ...) are ignored.

        ``checkpoint_num_layers`` records the file's ``num_layers`` before the overrides, so a
        ``num_layers=N`` override builds the first N blocks of the same checkpoint.
        """
        raw = json.loads((Path(transformer_path) / "config.json").read_text())
        names = {f.name for f in fields(cls)}
        kept = {k: v for k, v in raw.items() if k in names}
        if "axes_dims_rope" in kept:
            kept["axes_dims_rope"] = tuple(int(d) for d in kept["axes_dims_rope"])
        if "in_channels" in kept and kept.get("out_channels") is None:
            # Qwen-Image packs 2x2 latent patches: in_channels = 4 * the latent channels.
            kept["out_channels"] = int(kept["in_channels"]) // 4
        if "num_layers" in kept:
            kept["checkpoint_num_layers"] = int(kept["num_layers"])
        kept.update(overrides)
        config = cls(**kept)
        config.validate()
        return config


__all__ = ["NeuronQwenImageConfig"]
