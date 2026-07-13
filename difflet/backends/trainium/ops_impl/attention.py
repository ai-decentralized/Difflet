"""Trainium attention op passthroughs."""

import math
import os

import torch
import torch.nn.functional as F

from nkilib.core.attention.attention_cte import attention_cte

try:
    from nkilib.experimental.attention.ring_attention_fwd import ring_attention_spmd_fwd

    _RING_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import guard
    ring_attention_spmd_fwd = None
    _RING_IMPORT_ERROR = exc


def attention(
    q,
    k,
    v,
    *,
    scale: float | None = None,
    causal: bool = False,
    attention_mask=None,
    bound_min=None,
    bound_max=None,
    tp_q: bool = False,
    tp_k: bool = False,
    tp_out: bool = False,
    **kwargs,
):
    # Contiguous-masked (lossless) flash path: a caller that resolved its mask to
    # attention_cte's per-query bound_min/bound_max range (via
    # ops_impl.mask_bounds.mask_to_contiguous_bounds, computed OUTSIDE the traced
    # forward) routes here — measured lossless (cosine 0.99986) and ~1.16x vs SDPA.
    # NOTE: bounds must be resolved at trace-build time, not inside the traced
    # forward — running mask_to_contiguous_bounds in-graph trips an XLA broadcast
    # error on some mask shapes, so there is deliberately no in-graph auto-route.
    # range_select requires scale==1.0, so pre-scale q here.
    if bound_min is not None or bound_max is not None:
        assert bound_min is not None and bound_max is not None, (
            "bound_min and bound_max must both be provided"
        )
        vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
        kernel = attention_cte[2] if vc_size == 2 else attention_cte
        s = 1.0 if scale is None else float(scale)
        q_scaled = (q.float() * s).to(q.dtype) if s != 1.0 else q
        return kernel(
            q_scaled,
            k,
            v,
            scale=1.0,
            causal_mask=causal,
            bound_min=bound_min,
            bound_max=bound_max,
            tp_q=tp_q,
            tp_k=tp_k,
            tp_out=tp_out,
            **kwargs,
        )

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


def ring_attention(q, k, v, *, scale: float, causal: bool = False):
    """Ring context-parallel self-attention via nkilib ring_attention_spmd_fwd.

    q,k,v: [B, H, S_local, d] (per-rank head shard). Returns [B, H, S_local, d].
    The ring membership IS the cp-axis subgroup the model scattered Q with,
    so K/V rotate consistently with the scatter by construction.
    """
    if ring_attention_spmd_fwd is None:
        raise RuntimeError(
            "ring attention requires nkilib.experimental.attention.ring_attention_fwd "
            f"(import failed: {_RING_IMPORT_ERROR!r}). Upgrade neuronx-cc / nkilib, or "
            "use cp_mode=gather_kv."
        )
    from difflet.backends.trainium.core.parallel_mesh import get_cp_mesh

    mesh = get_cp_mesh()  # List[List[int]] of global ranks, one ring per cp group
    num_workers = len(mesh[0])
    replica_groups = tuple(tuple(int(r) for r in grp) for grp in mesh)

    # Launch with the LNC2 1D SPMD grid under virtual-core-size 2 (same as
    # attention_cte[2] above). The kernel's per-core shared_hbm send/recv buffers
    # and core_barrier require the grid; without it neuronx-cc fails to resolve
    # the named buffers on core 1 ("NCC_ILLC059 ... send_k_buf on core 1").
    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    kernel = ring_attention_spmd_fwd[2] if vc_size == 2 else ring_attention_spmd_fwd

    return kernel(
        q,
        k,
        v,
        replica_groups=replica_groups,
        num_workers=num_workers,
        softmax_scale=float(scale),
        use_causal_mask=causal,
        training=False,
        tp_q=True,
        tp_k=True,
    )


def _cte_stat_to_per_query(stat, bs: int, s_q: int):
    """Map an attention_cte cache_softmax stat to a per-query column vector.

    attention_cte (cache_softmax=True) returns softmax stats shaped
    [bs, 128, num_grps] where num_grps = ceil(s_q / 128). This is NKI's Q-group
    tile layout: partition row ``p`` in [0,128) and group ``g`` in [0,num_grps)
    address query position ``q = g*128 + p``. Transposing the last two dims
    ([bs, num_grps, 128]) then flattening gives the natural per-query order
    (g*128 + p); slicing to s_q drops the tail padding when s_q % 128 != 0.

    Confirmed on device by the Task 3 spike (Approach A): with this mapping the
    multi-partial online-softmax merge is lossless vs a full gather-KV joint
    attention (cosine >= 0.999).
    """
    return stat.transpose(-1, -2).reshape(bs, -1, 1)[:, :s_q, :].float()


def _merge_unnormalized_partials(partials, bs: int, s_q: int, *, out_dtype):
    """Online-softmax merge of N unnormalized attention_cte partials.

    Each partial is the (out, neg_max, sum) triple returned by attention_cte with
    cache_softmax=True AND skip_output_normalization=True, i.e.:
        out_i     [bs, s_q, d]          = exp(scores_i - max_i) @ V_i   (UNnormalized)
        neg_max_i [bs, 128, num_grps]   = -max_i  (negated per-query running max)
        sum_i     [bs, 128, num_grps]   = sum_j exp(scores_ij - max_i)  (raw denom)

    Combine via the standard flash/online-softmax rescale to the global max:
        nm*     = min_i neg_max_i               (= -global_max)
        corr_i  = exp(nm* - neg_max_i)          (= exp(max_i - global_max), in (0,1])
        out     = (sum_i corr_i * out_i) / (sum_i corr_i * sum_i)
    Order-invariant over partials, so the text partial and each ring-hop image
    partial merge regardless of arrival order (valid for non-causal joint attn).
    """
    neg_maxes = [_cte_stat_to_per_query(nm, bs, s_q) for (_, nm, _) in partials]
    sums = [_cte_stat_to_per_query(sm, bs, s_q) for (_, _, sm) in partials]

    global_neg_max = neg_maxes[0]
    for nm in neg_maxes[1:]:
        global_neg_max = torch.minimum(global_neg_max, nm)

    o_acc = None
    s_acc = None
    for (o, _, _), nm, sm in zip(partials, neg_maxes, sums):
        corr = torch.exp(global_neg_max - nm)  # [bs, s_q, 1]
        o_term = o.float() * corr
        s_term = sm * corr
        o_acc = o_term if o_acc is None else o_acc + o_term
        s_acc = s_term if s_acc is None else s_acc + s_term

    return (o_acc / s_acc).to(out_dtype)


def joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale: float, causal: bool = False):
    """Joint-MMDiT ring self-attention (Approach A — device-validated, Task 3).

    Approach B (reuse ``ring_attention_spmd_fwd`` for the image stream and merge a
    single text partial) is fundamentally rejected: that kernel reshapes K with the
    Q seqlen, so it requires ``seqlen_q == seqlen_k`` and cannot accept the joint
    query (``S_img/cp + S_txt``) against an image-only K shard (``S_img/cp``).

    Instead we hand-roll the context-parallel ring at the XLA level:
      * The replicated text K,V form ONE rank-local ``attention_cte`` partial.
      * The sharded image K,V are rotated around the cp ring with
        ``collective_permute``; every hop is another ``attention_cte`` partial.
      * Each ``attention_cte`` runs with cache_softmax=True + skip_output_norm=True
        so it returns the UNnormalized output and raw softmax stats; all
        ``num_workers + 1`` partials are merged once by online softmax.
    Every ``attention_cte`` natively supports ``seqlen_q != seqlen_k``, so the long
    joint query is fine. Lossless vs gather-KV (cosine >= 0.999 on device).

    q                [B, H, S_img/cp + S_txt, d]   this rank's joint local queries
    image_k,image_v  [B, H, S_img/cp, d]           sharded — rotated by the ring
    text_k, text_v   [B, H, S_txt,    d]           replicated — local partial
    returns          [B, H, S_img/cp + S_txt, d]
    """
    if causal:
        raise NotImplementedError(
            "joint_ring_attention device path supports only non-causal joint MMDiT "
            "attention (image+text bidirectional). Causal across ring-sharded image "
            "keys would need per-hop cp_offset masking, which no joint caller uses."
        )

    import torch_xla.core.xla_model as xm

    from difflet.backends.trainium.core.parallel_mesh import get_cp_mesh

    b, h, s_q, d = q.shape
    bs = b * h
    s_img = image_k.shape[2]
    s_txt = text_k.shape[2]

    qf = q.reshape(bs, s_q, d)
    ik = image_k.reshape(bs, s_img, d)
    iv = image_v.reshape(bs, s_img, d)
    tk = text_k.reshape(bs, s_txt, d)
    tv = text_v.reshape(bs, s_txt, d)

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    cte = attention_cte[2] if vc_size == 2 else attention_cte

    mesh = get_cp_mesh()  # List[List[int]] of global ranks, one ring per cp group
    num_workers = len(mesh[0])

    # collective_permute ring step: each rank sends its current K,V to the next
    # member of its cp group (g[i] -> g[i+1]), so after a hop every rank holds the
    # shard from its ring predecessor. num_workers-1 hops visit all image shards.
    pairs = []
    for grp in mesh:
        g = [int(r) for r in grp]
        n = len(g)
        for i in range(n):
            pairs.append([g[i], g[(i + 1) % n]])

    def _partial(k_in, v_in):
        # UNnormalized output + raw softmax stats for cross-partial online merge.
        return cte(
            qf, k_in, v_in,
            scale=float(scale), causal_mask=False,
            tp_q=True, tp_k=True, tp_out=False,
            cache_softmax=True, skip_output_normalization=True,
        )

    # Text partial: replicated, counted exactly once.
    partials = [_partial(tk, tv)]

    # Image partials: one per ring hop.
    k_cur, v_cur = ik, iv
    for step in range(num_workers):
        partials.append(_partial(k_cur, v_cur))
        if step < num_workers - 1:
            k_cur = xm.collective_permute(k_cur, pairs)
            v_cur = xm.collective_permute(v_cur, pairs)

    out = _merge_unnormalized_partials(partials, bs, s_q, out_dtype=q.dtype)
    return out.reshape(b, h, s_q, d)


def _cp_all_to_all(t, *, split_dim: int, concat_dim: int, mesh):
    """XLA AllToAll along the cp axis.

    Splits ``t`` into ``cp`` chunks along ``split_dim`` (chunk ``j`` goes to the
    ``j``-th member of the rank's cp group) and concatenates the ``cp`` received
    chunks along ``concat_dim`` **in replica-group order**. That ordering is what
    makes the Ulysses layout swap correct with no rank-dependent indexing.
    """
    import torch_xla.core.xla_model as xm

    return xm.all_to_all(
        t,
        split_dimension=split_dim,
        concat_dimension=concat_dim,
        split_count=len(mesh[0]),
        groups=[list(g) for g in mesh],
        # pin_layout=False is REQUIRED, not a tuning knob. With torch_xla's default
        # (True) the op is emitted as a layout-pinned CustomCall, and neuronx-cc
        # rejects the graph outright: "CustomCallOp unsupported target:
        # mhlo.all_to_all". Unpinned it lowers to a native HLO AllToAll, which the
        # compiler handles. This is what nxd's own expert-parallel all-to-all does
        # (parallel_layers/mappings.py::_all_to_all_in_expert_parallel_region).
        pin_layout=False,
    )


def _ulysses_check_heads(h: int, cp: int) -> None:
    if h % cp != 0:
        raise ValueError(
            f"cp_mode='ulysses' needs the per-rank head count ({h}) divisible by "
            f"cp_degree ({cp}): Ulysses shards heads across the cp axis on top of the "
            f"TP head shard, so num_attention_heads must be divisible by tp_degree * "
            f"cp_degree. Reduce cp_degree, or use cp_mode=gather_kv/ring."
        )


def _dense_attention(q, k, v, *, scale: float, causal: bool):
    """One ordinary dense attention over [B, H, S, d] tensors (no CP collectives)."""
    b, h, s_q, d = q.shape
    s_k = k.shape[2]
    bs = b * h
    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    kernel = attention_cte[2] if vc_size == 2 else attention_cte
    out = kernel(
        q.reshape(bs, s_q, d),
        k.reshape(bs, s_k, d),
        v.reshape(bs, s_k, d),
        scale=float(scale),
        causal_mask=causal,
        tp_q=True,
        tp_k=True,
        tp_out=False,
    )
    return out.reshape(b, h, s_q, d)


def ulysses_attention(q, k, v, *, scale: float, causal: bool = False):
    """Ulysses (all-to-all) context-parallel self-attention over a sharded sequence.

    q,k,v: [B, H_local, S/cp, d] → returns [B, H_local, S/cp, d].

    An all-to-all trades the sequence shard for a head shard ([B, H_local/cp, S, d]),
    so one *ordinary* dense attention sees the whole sequence — no ring kernel, no
    online-softmax merge, and no experimental nkilib dependency. A second all-to-all
    restores the caller's layout. Exact: the softmax is a single dense pass over the
    full sequence, identical math to gather-KV.

    Correctness rests on XLA AllToAll's ordering: on the forward hop rank ``r``
    receives head-block ``r`` from every rank ``r'``, concatenated along seq in group
    order — and rank ``r'`` holds sequence block ``r'``, so the concatenation lands in
    true global sequence order. The inverse hop restores head order by the same
    argument. Both are rank-agnostic, so the traced SPMD graph is identical on every
    rank.
    """
    if causal:
        # Not merely unimplemented — silently WRONG for at least one caller. Flux
        # shards a joint [text ‖ image] sequence, so the ranks' blocks concatenate as
        # [txt_0‖img_0, txt_1‖img_1, ...]: the all-to-all reconstructs the full token
        # set, but not in global sequence order. Non-causal attention is
        # permutation-invariant over keys so that is fine today; a causal mask would
        # be applied against the wrong positions. Every caller passes causal=False.
        raise NotImplementedError(
            "ulysses_attention supports only non-causal attention: the all-to-all "
            "reconstructs the full token set but not necessarily in global sequence "
            "order (flux shards a joint [text ‖ image] sequence), so a causal mask "
            "would be applied against the wrong positions."
        )
    from difflet.backends.trainium.core.parallel_mesh import get_cp_mesh

    mesh = get_cp_mesh()
    cp = len(mesh[0])
    _ulysses_check_heads(q.shape[1], cp)

    # heads → seq: [B, H_local, S/cp, d] → [B, H_local/cp, S, d]
    q = _cp_all_to_all(q, split_dim=1, concat_dim=2, mesh=mesh)
    k = _cp_all_to_all(k, split_dim=1, concat_dim=2, mesh=mesh)
    v = _cp_all_to_all(v, split_dim=1, concat_dim=2, mesh=mesh)

    out = _dense_attention(q, k, v, scale=scale, causal=causal)

    # seq → heads: [B, H_local/cp, S, d] → [B, H_local, S/cp, d]
    return _cp_all_to_all(out, split_dim=2, concat_dim=1, mesh=mesh)


def joint_ulysses_attention(
    q_img, q_txt, image_k, image_v, text_k, text_v, *, scale: float, causal: bool = False
):
    """Joint-MMDiT Ulysses attention: sharded image stream + replicated text stream.

    q_img/image_k/image_v  [B, H_local, S_img/cp, d]  sequence-sharded over cp
    q_txt/text_k/text_v    [B, H_local, S_txt,    d]  replicated on every rank
    returns (img_out [B, H_local, S_img/cp, d], txt_out [B, H_local, S_txt, d])

    The image stream takes the ordinary Ulysses all-to-all. The text stream is
    *replicated*, so it cannot be all-to-all'd for real — but each rank still needs
    the text restricted to its own head block. Applying the SAME all-to-all to it does
    exactly that for free: since every rank holds identical text, rank ``r`` receives
    ``cp`` **identical** copies of head-block ``r``, concatenated along seq into
    [B, H_local/cp, cp*S_txt, d]. Taking the first S_txt of that is this rank's head
    block — a static, rank-agnostic slice, so no SPMDRank plumbing is needed in the
    two models that hit this path.

    With both streams on the same head block and the image at full sequence length, a
    single dense attention over the concatenated [image ‖ text] keys is the exact
    joint result. On the way out the image half takes the inverse all-to-all back to
    its sequence shard, while the text half is all-gathered along heads to restore the
    replication its callers expect.
    """
    if causal:
        raise NotImplementedError(
            "joint_ulysses_attention supports only non-causal joint MMDiT attention "
            "(image+text bidirectional); no joint caller uses a causal mask."
        )
    from neuronx_distributed.parallel_layers.mappings import (
        gather_from_tensor_model_parallel_region_with_dim,
    )

    from difflet.backends.trainium.core.parallel_mesh import get_cp_group, get_cp_mesh

    mesh = get_cp_mesh()
    cp = len(mesh[0])
    _ulysses_check_heads(q_img.shape[1], cp)
    s_txt = q_txt.shape[2]

    def to_head_shard(sharded, replicated):
        """(seq-sharded → full-seq, replicated → this rank's head block)."""
        full = _cp_all_to_all(sharded, split_dim=1, concat_dim=2, mesh=mesh)
        # cp identical copies of our head block, concatenated along seq — keep one.
        tiled = _cp_all_to_all(replicated, split_dim=1, concat_dim=2, mesh=mesh)
        return full, tiled.narrow(2, 0, s_txt)

    q_i, q_t = to_head_shard(q_img, q_txt)
    k_i, k_t = to_head_shard(image_k, text_k)
    v_i, v_t = to_head_shard(image_v, text_v)

    # Image-first joint order here is internal only — each stream is handed back
    # separately, so callers keep their own [img‖txt] / [txt‖img] convention.
    q = torch.cat([q_i, q_t], dim=2)
    k = torch.cat([k_i, k_t], dim=2)
    v = torch.cat([v_i, v_t], dim=2)

    out = _dense_attention(q, k, v, scale=scale, causal=False)

    s_img = q_i.shape[2]
    img_out = out.narrow(2, 0, s_img)   # [B, H_local/cp, S_img, d]
    txt_out = out.narrow(2, s_img, s_txt)  # [B, H_local/cp, S_txt, d]

    # Image: back to this rank's sequence shard, full head count.
    img_out = _cp_all_to_all(img_out, split_dim=2, concat_dim=1, mesh=mesh)
    # Text: re-replicate across the cp group by gathering the head blocks back.
    txt_out = gather_from_tensor_model_parallel_region_with_dim(
        txt_out, gather_dim=1, process_group=get_cp_group()
    )
    return img_out, txt_out


__all__ = [
    "attention",
    "cross_attention",
    "ring_attention",
    "joint_ring_attention",
    "ulysses_attention",
    "joint_ulysses_attention",
]
