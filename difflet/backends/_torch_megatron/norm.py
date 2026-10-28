"""Normalization layers for the torch backends (TPU, GPU).

Megatron does not shard normalization: every rank keeps a full copy and
normalizes over the (replicated) hidden dim, so these are plain torch modules
with no collectives. Kept byte-compatible with the CPU backend's math so the
CPU reference stays a valid oracle for numerical validation.
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


def rms_norm_fp32(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim with fp32 statistics, fp32 normalisation and fp32 affine,
    the result cast back to ``x.dtype`` — diffusers' ``torch.nn.RMSNorm`` numerics, and the
    op ``WanAttention``'s across-heads q/k norm reduces to at tp=1. This is
    the reference: the same ops in the same order on every backend that has no fused
    kernel, so the CPU oracle, Trainium and TPU keep today's bits."""
    sq = x.float().pow(2).sum(dim=-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(sq / x.shape[-1] + eps)
    if weight is not None:
        x_normed = x_normed * weight.float()
    return x_normed.to(x.dtype)


__all__ = ["CustomRMSNorm", "LayerNorm", "RMSNorm", "rms_norm_fp32"]
