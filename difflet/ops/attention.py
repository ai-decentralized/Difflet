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


def joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale: float, causal: bool = False):
    """Joint-MMDiT ring self-attention for text-replicated models.

    Rings the sequence-sharded image K,V and merges a replicated text partial via
    online softmax. ``q`` is this rank's joint local query
    ``[B, H, S_img/cp + S_txt, d]``; ``image_*`` are the sharded image K,V;
    ``text_*`` are the replicated text K,V. Returns ``[B, H, q.shape[2], d]``.
    """

    return _load("joint_ring_attention")(
        q, image_k, image_v, text_k, text_v, scale=scale, causal=causal
    )


def ulysses_attention(q, k, v, *, scale: float, causal: bool = False):
    """Context-parallel all-to-all (Ulysses) self-attention over a sequence-sharded Q/K/V.

    ``q,k,v`` are ``[B, H_local, S/cp, d]`` (this rank's head shard, sequence-sharded
    over the cp axis). An all-to-all re-lays them out as ``[B, H_local/cp, S, d]``
    — full sequence, fewer heads — so a single *ordinary dense* attention computes the
    exact result; a second all-to-all restores ``[B, H_local, S/cp, d]``.

    Exact by construction (one dense softmax over the full sequence, same math as
    gather-KV) and needs no special attention kernel. Requires ``H_local % cp == 0``.
    """

    return _load("ulysses_attention")(q, k, v, scale=scale, causal=causal)


def joint_ulysses_attention(
    q_img, q_txt, image_k, image_v, text_k, text_v, *, scale: float, causal: bool = False
):
    """Joint-MMDiT Ulysses self-attention for text-replicated models.

    The image stream is sequence-sharded over the cp axis while the text stream is
    replicated on every rank, so the two cannot share one pre-concatenated query the
    way ``joint_ring_attention`` takes one — they are passed separately and the
    ``(img_out, txt_out)`` pair is returned separately, letting each caller reassemble
    in its own joint order (hunyuan concatenates ``[img ‖ txt]``, qwen ``[txt ‖ img]``).

    ``q_img``/``image_k``/``image_v`` are ``[B, H_local, S_img/cp, d]`` (sharded);
    ``q_txt``/``text_k``/``text_v`` are ``[B, H_local, S_txt, d]`` (replicated).
    Returns ``img_out`` ``[B, H_local, S_img/cp, d]`` and ``txt_out``
    ``[B, H_local, S_txt, d]`` — i.e. each stream in the sharding it arrived with.
    """

    return _load("joint_ulysses_attention")(
        q_img, q_txt, image_k, image_v, text_k, text_v, scale=scale, causal=causal
    )


def _load(name: str):
    from difflet.ops._dispatch import load_backend_attr

    return load_backend_attr("attention", name)


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'difflet.ops.attention' has no attribute {name!r}")
    return _load(name)
