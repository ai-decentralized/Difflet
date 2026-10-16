"""Normalization layers for TPU.

Megatron does not shard normalization: every rank keeps a full copy and
normalizes over the (replicated) hidden dim, so these are plain torch modules
with no collectives. Kept byte-compatible with the CPU backend's math so the
CPU reference stays a valid oracle for numerical validation (Phase 5).

Phase 2b of docs/plans/2026-08-16-tpu-backend-support.md.
"""

# Lifted into difflet.backends._torch_megatron (CUDA plan D10); re-exported here so
# the TPU backend and its tests keep their import paths.
from difflet.backends._torch_megatron.norm import (  # noqa: F401
    LayerNorm,
    RMSNorm,
    CustomRMSNorm,
)

__all__ = ["CustomRMSNorm", "LayerNorm", "RMSNorm"]
