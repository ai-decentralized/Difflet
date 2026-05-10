"""Trainium attention op passthroughs."""

import os

from nkilib.core.attention.attention_cte import attention_cte


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
    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    kernel = attention_cte[2] if vc_size == 2 else attention_cte
    return kernel(
        q,
        k,
        v,
        scale=1.0 if scale is None else scale,
        causal_mask=causal,
        tp_q=tp_q,
        tp_k=tp_k,
        tp_out=tp_out,
        **kwargs,
    )


def cross_attention(q, k, v, *, scale: float | None = None, **kwargs):
    return attention(q, k, v, scale=scale, causal=False, tp_q=False, tp_k=False, tp_out=False, **kwargs)


__all__ = ["attention", "cross_attention"]
