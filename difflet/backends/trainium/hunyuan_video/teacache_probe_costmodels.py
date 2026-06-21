"""Measurement-only model variants for decomposing the TeaCache probe NEFF
per-step cost (cclog 79, Test 1). NOT for production use.

Three variants compiled as standalone NEFFs to isolate where the ~51 ms/step
probe overhead goes:

- DeltaOnly: same compute as the real probe but returns ONLY the scalar delta
  (drops the 63 MB mod_input output). probe_full − DeltaOnly = mod_input output
  write cost.
- Trivial: a near-empty graph (returns a reduction of a small input) to measure
  the pure NEFF dispatch + mark_step floor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from difflet.models.hunyuan_video.modeling_hunyuan_video import (
    HunyuanVideoTransformer3DModel,
)


class HunyuanVideoTeacacheProbeDeltaOnly(nn.Module):
    """Probe variant returning only the scalar delta (no mod_input tensor)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.model = HunyuanVideoTransformer3DModel(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
        prev_mod_input: torch.Tensor,
    ) -> torch.Tensor:
        mod_input = self.model.teacache_mod_input(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
        )
        diff = (mod_input - prev_mod_input).reshape(-1)
        return torch.linalg.vector_norm(diff)


class HunyuanVideoTrivialNEFF(nn.Module):
    """Near-empty graph: a sum reduction over a small input. Measures the pure
    dispatch + mark_step floor with negligible compute."""

    def __init__(self, config) -> None:
        super().__init__()
        # one tiny parameter so the module is non-empty for the builder
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.reshape(-1).sum() * self.scale).reshape(())
