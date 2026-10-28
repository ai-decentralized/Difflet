"""Normalization layers for the CUDA backend.

Megatron does not shard normalization, so these are plain modules with no collectives.
``LayerNorm`` is the shared pure-torch one.

``rms_norm_fp32`` and ``RMSNorm`` are one fused ``torch.nn.functional.rms_norm`` call: fp32
statistics, normalisation and affine inside one kernel, one cast back to ``x.dtype``, instead
of the reference's seven elementwise passes over fp32 copies (a measurable share of Wan's and
Flux's DiT time).

Numerics against the CPU reference: ``rms_norm_fp32`` is the reference's math with a
different fp32 reduction order. ``RMSNorm`` is not the CPU ``RMSNorm``'s math: the CPU module
normalises and applies the weight in the input dtype, this one computes in fp32 like
diffusers' ``nn.RMSNorm``; the two agree within bf16 rounding, not bit for bit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends._torch_megatron.norm import LayerNorm  # noqa: F401


def rms_norm_fp32(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=float(eps))


class RMSNorm(nn.Module):
    """The module behind ``difflet.ops.RMSNorm`` / ``CustomRMSNorm`` on CUDA: one fused
    ``F.rms_norm`` — fp32 statistics, normalisation and affine, one cast to ``x.dtype`` —
    diffusers' ``nn.RMSNorm`` numerics. The shared pure-torch module normalises in the input
    dtype over several passes; Flux's and Qwen-Image's q/k norms are this module, and the Flux
    DiT bench put the gap to diffusers in exactly those passes. Same ``weight`` parameter and
    ``eps`` as the shared module, so checkpoints load unchanged.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        del kwargs
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm_fp32(hidden_states, self.weight, self.eps)


CustomRMSNorm = RMSNorm

__all__ = ["CustomRMSNorm", "LayerNorm", "RMSNorm", "rms_norm_fp32"]
