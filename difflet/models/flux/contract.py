"""Backend-neutral host-side contract for one Flux DiT call (CUDA plan F5).

The Trainium application keeps its own NxD-bound input handling; this module is pure — a
tensor dataclass and a dtype mapping — so the CUDA application (and anything else off
Trainium) can describe a DiT call without importing the Neuron toolchain. Modelled on
``difflet.models.qwen_image.contract``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FluxDiTInputBundle:
    """One ``NeuronFluxTransformer2DModel`` call, in the DiT's positional order."""

    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    pooled_projections: torch.Tensor
    timestep: torch.Tensor
    guidance: torch.Tensor
    image_rotary_emb: torch.Tensor

    def as_model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.encoder_hidden_states,
            self.pooled_projections,
            self.timestep,
            self.guidance,
            self.image_rotary_emb,
        )


def normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported Flux dtype: {dtype!r}")


__all__ = ["FluxDiTInputBundle", "normalize_dtype"]
