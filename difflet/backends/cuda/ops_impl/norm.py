"""Normalization layers for the CUDA backend: the shared pure-torch ones.

Megatron does not shard normalization, so these are plain modules with no
collectives, byte-identical in math to the CPU reference (the oracle).
"""

from difflet.backends._torch_megatron.norm import (  # noqa: F401
    CustomRMSNorm,
    LayerNorm,
    RMSNorm,
)

__all__ = ["CustomRMSNorm", "LayerNorm", "RMSNorm"]
