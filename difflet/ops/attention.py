"""Attention operations."""


def attention(
    q,
    k,
    v,
    *,
    scale: float | None = None,
    causal: bool = False,
    attention_mask=None,
    tp_q: bool = False,
    tp_k: bool = False,
    tp_out: bool = False,
    **kwargs,
):
    """Backend attention entry point.

    Public model code should call this shape, not backend-specific kernels.
    The Trainium implementation maps it to nkilib ``attention_cte``.
    """

    return _load("attention")(
        q,
        k,
        v,
        scale=scale,
        causal=causal,
        attention_mask=attention_mask,
        tp_q=tp_q,
        tp_k=tp_k,
        tp_out=tp_out,
        **kwargs,
    )


def cross_attention(q, k, v, *, scale: float | None = None, attention_mask=None, **kwargs):
    return attention(
        q,
        k,
        v,
        scale=scale,
        causal=False,
        attention_mask=attention_mask,
        tp_q=False,
        tp_k=False,
        tp_out=False,
        **kwargs,
    )


def ring_attention(q, k, v, *, scale: float, causal: bool = False):
    """Context-parallel ring self-attention over a sequence-sharded Q/K/V.

    ``q,k,v`` are ``[B, H, S_local, d]`` (this rank's head shard). The backend
    resolves the data-parallel ring group and merges per-step partials.
    """

    return _load("ring_attention")(q, k, v, scale=scale, causal=causal)


def _load(name: str):
    from difflet.ops._dispatch import load_backend_attr

    return load_backend_attr("attention", name)


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'difflet.ops.attention' has no attribute {name!r}")
    return _load(name)
