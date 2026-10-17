"""Shape and topology config for the Wan DiT on the neuron backend.

A copy of ``TpuWanConfig`` (``difflet/backends/tpu/wan/config.py:29-133``), not an import, so
the two backends can diverge without editing the TPU tree: ``WanTransformerConfig`` (the
hardware-neutral hyperparameters, ``difflet/models/wan/modeling_wan.py``) plus the one static
runtime shape, the tp degree and the dtype, and the latent geometry the warm-up inputs are
built from. ``validate()`` differs in the parallel modes: it checks their divisibility (CP's
sequence split and Ulysses' head split, SP's sequence split over TP) the way
``NeuronQwenImageConfig`` does, and the ``num_layers`` cut against ``checkpoint_num_layers``.

The parallel-mode flags are the ones ``WanTransformer3DModel`` reads with ``getattr``; they
are declared so that an unsupported or ragged mode fails here, with a message, instead of
building a model that ignores it or fails in its first forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from difflet.models.wan.modeling_wan import WanTransformerConfig
from difflet.pipeline.parallel_config import CP_MODES


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
        if int(self.num_layers) < 1:
            raise ValueError(f"Wan num_layers={self.num_layers} must be >= 1")
        if self.checkpoint_num_layers is not None and int(self.num_layers) > int(
            self.checkpoint_num_layers
        ):
            raise ValueError(
                f"Wan num_layers={self.num_layers} exceeds the checkpoint's "
                f"{self.checkpoint_num_layers} blocks"
            )
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
        if self.cp_mode not in CP_MODES:
            raise ValueError(f"cp_mode must be one of {CP_MODES}, got {self.cp_mode!r}")
        if self.context_parallel_enabled and self.sp_enabled:
            raise ValueError(
                "Wan sp_enabled and context_parallel_enabled are mutually exclusive (both shard "
                "the sequence dimension)"
            )
        if self.context_parallel_enabled and self.cfg_parallel_enabled:
            raise ValueError(
                "Wan cfg_parallel_enabled and context_parallel_enabled are mutually exclusive "
                "(both consume the data-parallel lanes)"
            )
        heads, tp = int(self.num_attention_heads), int(self.tp_degree)
        seq = int(self.image_seq_len)
        if self.sp_enabled and seq % tp:
            raise ValueError(
                f"Wan sequence parallelism splits image_seq_len={seq} over tp={tp}, which does "
                "not divide it"
            )
        cp = None if self.cp_degree is None else int(self.cp_degree)
        if cp is not None and cp > 1 and not self.context_parallel_enabled:
            raise ValueError(
                f"cp_degree={cp} needs context_parallel_enabled=True (the module would run "
                "the full sequence on every rank of the CP group)"
            )
        if not self.context_parallel_enabled or cp is None or cp <= 1:
            return
        if seq % cp:
            raise ValueError(
                f"Wan image_seq_len={seq} ({self.latent_frames}x{self.latent_height}x"
                f"{self.latent_width} latent, patch {tuple(self.patch_size)}) is not divisible "
                f"by cp_degree={cp}"
            )
        if self.cp_mode == "ulysses" and (heads // tp) % cp:
            raise ValueError(
                f"cp_mode='ulysses' shards this rank's {heads // tp} heads (heads={heads} / "
                f"tp={tp}) over cp_degree={cp}; num_attention_heads must be divisible by "
                "tp_degree * cp_degree"
            )

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
