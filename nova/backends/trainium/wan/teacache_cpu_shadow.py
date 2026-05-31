"""Wan 2.2 TeaCache CPU shadow (cclog 87).

Computes the block-0 modulated-input TeaCache signal on the HOST (CPU), mirroring
``hunyuan_video/teacache_cpu_shadow.py``. Wan's block-0 AdaLN modulation is
timestep-only, but the gate measured Pearson 0.99 (the modulated input carries the
evolving latent), so the relative-L1 step-to-step change of this tensor is a strong
adaptive-skip signal.

Lean by construction: it builds the transformer with ``num_layers=1`` (only block-0 is
needed for ``teacache_mod_input``) and loads ONLY the ``patch_embedding`` /
``condition_embedder`` / ``blocks.0`` weights from the sharded checkpoint, so the host
footprint is a few hundred MB rather than the full ~28 GB expert.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import torch

from nova.models.wan.modeling_wan import WanTransformer3DModel, WanTransformerConfig

# Only the layers ``teacache_mod_input`` actually reads: patch_embedding,
# condition_embedder (timestep->timestep_proj), block-0 norm1 + scale_shift_table.
# (block-0 attn/ffn are constructed but never called, and the Nova FFN module names
# differ from the diffusers checkpoint anyway, so we don't load them.)
_NEEDED_PREFIXES = (
    "patch_embedding.",
    "condition_embedder.",
    "blocks.0.norm1.",
    "blocks.0.scale_shift_table",
)


class WanTeacacheCPUShadow:
    """Host-side block-0 modulated-input computer for one Wan transformer (stage)."""

    def __init__(self, transformer_path: str, dtype: torch.dtype = torch.bfloat16) -> None:
        config_path = os.path.join(transformer_path, "config.json")
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        config = WanTransformerConfig.from_diffusers_dict(raw)
        config.num_layers = 1  # only block-0 participates in teacache_mod_input
        self.dtype = dtype
        model = WanTransformer3DModel(config)
        self._load_needed_weights(model, transformer_path)
        self.model = model.to(dtype=dtype).eval()

    @staticmethod
    def _load_needed_weights(model: torch.nn.Module, transformer_path: str) -> None:
        from safetensors import safe_open

        shards = sorted(glob.glob(os.path.join(transformer_path, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no safetensors under {transformer_path}")
        state: dict[str, torch.Tensor] = {}
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith(_NEEDED_PREFIXES):
                        state[k] = f.get_tensor(k)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # blocks.1.. and tail layers are intentionally absent (num_layers=1); only the
        # block-0 / embedder weights must be present.
        needed_missing = [m for m in missing if m.startswith(_NEEDED_PREFIXES)]
        if needed_missing:
            raise RuntimeError(f"Wan shadow missing required weights: {needed_missing[:6]}")

    @torch.no_grad()
    def teacache_mod_input(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.teacache_mod_input(
            hidden_states.detach().to(dtype=self.dtype, device="cpu"),
            timestep.detach().to(device="cpu"),
            encoder_hidden_states.detach().to(dtype=self.dtype, device="cpu"),
        )
