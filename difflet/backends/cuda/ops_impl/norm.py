"""Normalization layers for the CUDA backend: the shared pure-torch ones.

Megatron does not shard normalization, so these are plain modules with no
collectives, byte-identical in math to the CPU reference (the oracle).

``rms_norm_fp32`` is the one fused path: ``torch.nn.functional.rms_norm`` computes the
statistics, the normalisation and the affine in fp32 inside one kernel and writes ``x.dtype``
once, where the reference's seven elementwise passes over fp32 copies cost Wan's DiT ~50 ms
of its 605 ms forward (the DiT bench: elementwise +51 ms, copy +34 ms vs diffusers). The math is
the reference's; only the fp32 reduction order differs.
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
