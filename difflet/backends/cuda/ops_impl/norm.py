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
import torch.nn.functional as F

from difflet.backends._torch_megatron.norm import (  # noqa: F401
    CustomRMSNorm,
    LayerNorm,
    RMSNorm,
)


def rms_norm_fp32(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=float(eps))


__all__ = ["CustomRMSNorm", "LayerNorm", "RMSNorm", "rms_norm_fp32"]
