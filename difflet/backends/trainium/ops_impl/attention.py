"""Trainium attention op passthroughs."""

import math
import os

import torch
import torch.nn.functional as F

from nkilib.core.attention.attention_cte import attention_cte


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
    if attention_mask is not None:
        return _masked_sdpa_attention(
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


def _masked_sdpa_attention(
    q,
    k,
    v,
    *,
    scale: float | None,
    causal: bool,
    attention_mask,
    tp_q: bool,
    tp_k: bool,
    tp_out: bool,
    **kwargs,
):
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise NotImplementedError(
            "Trainium masked SDPA fallback does not support attention_cte-only "
            f"kwargs: {unsupported}"
        )

    q_sdpa = q if tp_q else q.transpose(-1, -2).contiguous()
    k_sdpa = k if tp_k else k.transpose(-1, -2).contiguous()
    desired_scale = 1.0 if scale is None else float(scale)
    default_scale = 1.0 / math.sqrt(q_sdpa.shape[-1])
    if desired_scale != default_scale:
        q_sdpa = q_sdpa * (desired_scale / default_scale)
    if causal:
        attention_mask = _merge_causal_mask(q_sdpa, k_sdpa, attention_mask)
        causal = False

    out = F.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=causal,
    )
    return out.transpose(-1, -2).contiguous() if tp_out else out


def _merge_causal_mask(q, k, attention_mask):
    q_len, k_len = q.shape[-2], k.shape[-2]
    causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
    if attention_mask.dtype == torch.bool:
        return attention_mask & causal_mask
    causal_bias = torch.zeros((q_len, k_len), dtype=q.dtype, device=q.device)
    causal_bias = causal_bias.masked_fill(~causal_mask, torch.finfo(q.dtype).min)
    return attention_mask.to(device=q.device, dtype=q.dtype) + causal_bias


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


__all__ = ["attention", "cross_attention"]
