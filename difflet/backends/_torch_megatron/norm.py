"""Normalization layers for TPU.

Megatron does not shard normalization: every rank keeps a full copy and
normalizes over the (replicated) hidden dim, so these are plain torch modules
with no collectives. Kept byte-compatible with the CPU backend's math so the
CPU reference stays a valid oracle for numerical validation (Phase 5).

Phase 2b of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LayerNorm = nn.LayerNorm


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        del kwargs
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # fp32 variance regardless of input dtype: bf16 accumulation here is a
        # known accuracy trap and the CPU oracle does the same.
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance.to(hidden_states.dtype) + self.eps)
        return hidden_states * self.weight


CustomRMSNorm = RMSNorm

__all__ = ["CustomRMSNorm", "LayerNorm", "RMSNorm"]
