"""Shape and topology config for the Wan DiT on the neuron backend.

A copy of ``TpuWanConfig`` (``difflet/backends/tpu/wan/config.py:29-133``), not an import, so
the two backends can diverge without editing the TPU tree: ``WanTransformerConfig`` (the
hardware-neutral hyperparameters, ``difflet/models/wan/modeling_wan.py``) plus the one static
runtime shape, the tp degree and the dtype, and the latent geometry the warm-up inputs are
built from. ``validate()`` differs in the parallel modes: it checks their divisibility (CP's
sequence split and Ulysses' head split, SP's sequence split over TP) and the ``num_layers`` cut
against ``checkpoint_num_layers`` with the rules every neuron DiT config shares
(``difflet/backends/neuron/core/config_checks.py``).

The parallel-mode flags are the ones ``WanTransformer3DModel`` reads with ``getattr``; they
are declared so that an unsupported or ragged mode fails here, with a message, instead of
building a model that ignores it or fails in its first forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from difflet.models.wan.modeling_wan import WanTransformerConfig


@dataclass
class NeuronWanConfig(WanTransformerConfig):
    """``WanTransformerConfig`` plus the neuron runtime's shape and topology."""

    #: The checkpoint's own block count, set by ``from_pretrained`` before a ``num_layers``
    #: override. An on-device parity check that builds only the first N blocks of the real
    #: checkpoint (``from_pretrained(..., num_layers=N)``, as for Qwen-Image) needs it to skip
    #: the rest: ``NeuronWanTransformerApplication`` reports them under ``load_report["skipped"]``
    #: rather than as unexpected.
    checkpoint_num_layers: int | None = None

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
    #: The CP degree when known (``NeuronWanTransformerApplication`` sets it from the parallel
    #: config); None leaves the cp-dependent checks to the mesh, which derives cp from the world
    #: size (``mesh_spec_from_config``).
    cp_degree: int | None = None
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
        check_layer_count("Wan", self.num_layers, self.checkpoint_num_layers)
        check_heads_divide_tp("Wan", self.num_attention_heads, self.tp_degree)
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
        self._validate_parallel()

    def _validate_parallel(self) -> None:
        """The parallel modes' own divisibility rules, mirroring ``NeuronQwenImageConfig``.

        ``WanTransformer3DModel`` splits the patch sequence with ``scatter_to_process_group_spmd``,
        which has no padding: CP splits it over the CP group, SP over the TP group, so a ragged
        split would only surface as a shape error deep in the first forward (or, under
        compile, in the block graph's trace). Ulysses then shards this rank's heads over CP.
        The exclusivity rules are the model's own (``WanTransformer3DModel.__init__``), checked
        here so that a wrong config fails before the mesh and the meta build.
        """
        check_cp_mode(self.cp_mode)
        if self.context_parallel_enabled and self.sp_enabled:
            raise exclusive_modes("Wan", "sp_enabled", "context_parallel_enabled",
                                  "both shard the sequence dimension")
        if self.context_parallel_enabled and self.cfg_parallel_enabled:
            raise exclusive_modes("Wan", "cfg_parallel_enabled", "context_parallel_enabled",
                                  "both consume the data-parallel lanes")
        heads, tp = int(self.num_attention_heads), int(self.tp_degree)
        seq = int(self.image_seq_len)
        if self.sp_enabled:
            check_even_split(f"Wan sequence parallelism splits image_seq_len={seq}", seq, "tp", tp)
        cp = None if self.cp_degree is None else int(self.cp_degree)
        check_cp_enabled(cp, self.context_parallel_enabled)
        if not self.context_parallel_enabled or cp is None or cp <= 1:
            return
        check_cp_split("Wan", "image_seq_len", seq,
                       f"{self.latent_frames}x{self.latent_height}x{self.latent_width} latent, "
                       f"patch {tuple(self.patch_size)}", cp)
        if self.cp_mode == "ulysses":
            check_ulysses_heads(heads, tp, cp)

    @classmethod
    def from_pretrained(cls, transformer_path, **overrides) -> "NeuronWanConfig":
        """Build from a diffusers ``transformer/config.json``.

        ``checkpoint_num_layers`` records the file's ``num_layers`` before the overrides, so a
        ``num_layers=N`` override builds the first N blocks of the same checkpoint.
        """
        raw = json.loads((Path(transformer_path) / "config.json").read_text())
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kept = {k: v for k, v in raw.items() if k in fields}
        if isinstance(kept.get("patch_size"), list):
            kept["patch_size"] = tuple(kept["patch_size"])
        if "num_layers" in kept:
            kept["checkpoint_num_layers"] = int(kept["num_layers"])
        kept.update(overrides)
        config = cls(**kept)
        config.validate()
        return config


__all__ = ["NeuronWanConfig"]
