"""Trainium gated linear attention — not yet implemented.

This module exists so that importing `difflet.ops.gla_attention` under the
Trainium backend fails with a clear NotImplementedError rather than a raw
ModuleNotFoundError from `difflet.ops._dispatch.load_backend_attr`, which only
converts a *missing attribute* into a friendly error, not a missing module.

The NKI kernel lands separately. Planned order: non-causal first (a single
reduction plus two matmuls, no chunk state chain), then causal (chunkwise with
inter-chunk state passing).
"""

from __future__ import annotations


def gla_attention(q, k, v, g, **kwargs):
    raise NotImplementedError(
        "gla_attention has no Trainium implementation yet. "
        "Set DIFFLET_BACKEND=cpu to use the reference implementation."
    )


__all__ = ["gla_attention"]
