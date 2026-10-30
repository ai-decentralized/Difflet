"""Rotary embedding: the CPU backend's pure-torch implementation, verified on the neuron device."""

from difflet.backends.cpu.ops_impl.embeddings import apply_rotary_emb

__all__ = ["apply_rotary_emb"]
