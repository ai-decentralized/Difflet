"""Attention operations."""


def attention(
    q,
    k,
    v,
    *,
    scale: float | None = None,
    causal: bool = False,
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
        tp_q=tp_q,
        tp_k=tp_k,
        tp_out=tp_out,
        **kwargs,
    )


def cross_attention(q, k, v, *, scale: float | None = None, **kwargs):
    return attention(q, k, v, scale=scale, causal=False, tp_q=False, tp_k=False, tp_out=False, **kwargs)


def attention_cte(*args, **kwargs):
    return _load("attention_cte")(*args, **kwargs)


def _load(name: str):
    from nova.ops._dispatch import load_backend_attr

    return load_backend_attr("attention", name)


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'nova.ops.attention' has no attribute {name!r}")
    return _load(name)
