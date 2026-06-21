"""HunyuanVideo TeaCache CPU shadow.

Computes the block-0 modulated input on the HOST (CPU) instead of via a
dedicated Trainium probe NEFF. Motivation (cclog 77): the probe NEFF costs
~51 ms/step of fixed dispatch + mark_step tax even though the underlying
compute is microseconds. On CPU the same three layers (x_embedder,
time_text_embed, transformer_blocks[0].norm1) run in ~single-digit ms with
no NeuronCore dispatch and no device synchronization.

The latent is already host-resident between denoise steps (the scheduler runs
on host), so the CPU shadow needs no extra device→host transfer.

This module only loads the three layers it needs, not the full transformer,
to keep the host RAM footprint small.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from difflet.models.hunyuan_video.modeling_hunyuan_video import (
    HunyuanVideoTransformer3DModel,
    HunyuanVideoTransformerConfig,
)


class HunyuanVideoTeacacheCPUShadow:
    """Host-side block-0 modulated-input computer.

    Holds a CPU copy of HunyuanVideoTransformer3DModel but only ever calls its
    ``teacache_mod_input`` path. (A future optimization can strip the unused
    layers; for now correctness + simplicity wins — the unused blocks sit idle
    in host RAM.)
    """

    def __init__(self, source_dir: str, dtype: torch.dtype = torch.bfloat16) -> None:
        transformer_path = os.path.join(source_dir, "transformer")
        config_path = os.path.join(transformer_path, "config.json")
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        config = HunyuanVideoTransformerConfig.from_diffusers_dict(raw)
        self.dtype = dtype
        model = HunyuanVideoTransformer3DModel(config)
        self._load_weights(model, transformer_path)
        self.model = model.to(dtype=dtype).eval()

    @staticmethod
    def _load_weights(model: torch.nn.Module, transformer_path: str) -> None:
        from safetensors.torch import load_file

        weights_path = os.path.join(
            transformer_path, "diffusion_pytorch_model.safetensors"
        )
        state = load_file(weights_path, device="cpu")
        # Only the teacache_mod_input path is exercised; load with strict=False
        # so a truncated/partial checkpoint still populates the needed layers.
        model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def teacache_mod_input(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.teacache_mod_input(
            hidden_states.detach().to(dtype=self.dtype, device="cpu"),
            timestep.detach().to(device="cpu"),
            encoder_hidden_states.detach().to(dtype=self.dtype, device="cpu"),
            encoder_attention_mask.detach().to(device="cpu"),
            pooled_projections.detach().to(dtype=self.dtype, device="cpu"),
            guidance.detach().to(dtype=self.dtype, device="cpu"),
        )
