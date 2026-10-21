"""Positional embedding helpers for TPU.

Identical math to the Trainium and CPU implementations — this is pure torch
with no backend-specific kernel, so it is reused verbatim rather than
rewritten. Keeping it byte-identical is deliberate: rotary embeddings are an
easy place for a silent interleaving bug, and divergence here would be
invisible until numerical validation.

Phase 2b of docs/plans/2026-08-16-tpu-backend-support.md.
"""

# Lifted into difflet.backends._torch_megatron; re-exported here so
# the TPU backend and its tests keep their import paths.
from difflet.backends._torch_megatron.embeddings import (  # noqa: F401
    apply_rotary_emb,
)

__all__ = ["apply_rotary_emb"]
