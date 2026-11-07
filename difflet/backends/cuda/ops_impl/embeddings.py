"""Rotary embeddings for the CUDA backend: the shared pure-torch implementation."""

from difflet.backends._torch_megatron.embeddings import (  # noqa: F401
    apply_rotary_emb,
    apply_rotary_emb_fp32,
)

__all__ = ["apply_rotary_emb", "apply_rotary_emb_fp32"]
