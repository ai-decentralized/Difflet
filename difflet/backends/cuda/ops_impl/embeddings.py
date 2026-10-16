"""Rotary embeddings for the CUDA backend: the shared pure-torch implementation."""

from difflet.backends._torch_megatron.embeddings import apply_rotary_emb  # noqa: F401

__all__ = ["apply_rotary_emb"]
