"""Backend-neutral host-side contract for one Qwen-Image transformer call.

Split out of ``application.py`` for the same reason the modeling was: that
module imports NxD at module level, so anything importing it drags the whole
Neuron toolchain in. These two pieces are pure — a tensor dataclass and a
dtype mapping — and both backends need them.

``application.py`` re-exports them, so existing Trainium importers are
unaffected.

Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class QwenImageDiTInputBundle:
    """Host-side contract for one Qwen-Image transformer call."""

    hidden_states: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_hidden_states_mask: torch.Tensor
    guidance: torch.Tensor

    def as_model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.timestep,
            self.encoder_hidden_states,
            self.encoder_hidden_states_mask,
            self.guidance,
        )


def normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported Qwen-Image dtype: {dtype!r}")


__all__ = ["QwenImageDiTInputBundle", "normalize_dtype"]
