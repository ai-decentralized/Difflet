"""Torch-native normalization layers for CPU numerical checks."""

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
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance.to(hidden_states.dtype) + self.eps)
        return hidden_states * self.weight


CustomRMSNorm = RMSNorm

__all__ = ["CustomRMSNorm", "LayerNorm", "RMSNorm"]
