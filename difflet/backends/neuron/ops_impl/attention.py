"""Attention on the neuron device: the NKI flash kernel, with SDPA as the fallback.

Unmasked inference attention calls torch-neuronx's NKI flash-attention kernel
directly. Through ``F.scaled_dot_product_attention`` the kernel is reached only
when both sequence lengths are multiples of 512; the kernel itself takes other
lengths (Wan: 4,680 tokens), so the direct call avoids the decomposed path.
The kernel computes in bf16 internally, so only bf16/fp16 inputs take it; fp32 keeps
fp32 precision through SDPA. Masked or autograd calls, and shapes outside the
kernel's limits, also use SDPA.

The attention_cte flash paths (unmasked, key-bounded, logsumexp) run it with
``allow_vector_exp=False`` by default: the softmax exp on the Scalar engine, which on trn3
takes it off the saturated Vector engine (see ``attention``'s ``vector_exp``). The unmasked
and logsumexp paths therefore call the kernel themselves (``_flash_fwd``), as
torch-neuronx's SDPA op would, since that op cannot pass the argument. Not covered:
``vector_exp=True`` calls, an nkilib without the argument (the previous kernels run), and
the NKI Library ring kernel (``_ring_kernel_op``, which has no such argument).

The entry points keep the attention_cte calling contract that model code is
written against:

* layout flags: ``tp_q``/``tp_k`` True means q/k are ``[..., S, d]``; False means
  ``[..., d, S]``. ``v`` is always ``[..., S_k, d]``. ``tp_out`` True returns
  ``[..., d, S_q]`` instead of ``[..., S_q, d]``.
* ``scale=None`` means 1.0 (not SDPA's 1/sqrt(d)).
* ``bound_min``/``bound_max``: per query row, keys in [bound_min, bound_max) are
  valid; mutually exclusive with ``attention_mask``.
* boolean masks are True where attention is allowed; float masks are additive.

Context parallelism: ``ulysses_attention`` swaps the sequence shard for a head shard around
plain attention; ``ring_attention`` passes K/V blocks around the CP group and merges the
partial results by their logsumexp, from the same flash kernel run with its softmax statistics.
The joint (text + image) variants serve MMDiT models whose image stream is sequence-sharded
over the CP group while the text stream is replicated on every rank (Qwen-Image, HunyuanVideo):
``joint_ring_attention`` folds the text K/V in as one local partial and rings only the image
K/V; ``joint_ulysses_attention`` swaps the image stream to a head shard, cuts the replicated
text to the same heads, and runs one dense attention over both. All four are non-causal only.
"""

from __future__ import annotations

import inspect
import math
import os

import torch
import torch.nn.functional as F

from difflet.backends.neuron.ops_impl.mx_attention import load_op, record_call
from difflet.quant.mx_attention import smooth_keys

#: ``DIFFLET_NEURON_ATTENTION=sdpa`` sends every call through ``F.scaled_dot_product_attention``
#: (never the direct kernel call): the A/B switch behind the SDPA-gate measurement. Read once at
#: import, so compiled regions see a constant. Default ``auto``: the direct kernel when eligible.
ATTENTION_PATH_ENV = "DIFFLET_NEURON_ATTENTION"
_ATTENTION_PATH = os.environ.get(ATTENTION_PATH_ENV, "auto").strip().lower()
if _ATTENTION_PATH not in ("auto", "sdpa"):
    raise ValueError(f"{ATTENTION_PATH_ENV} must be 'auto' or 'sdpa', got {_ATTENTION_PATH!r}")

#: ``DIFFLET_NEURON_RING_KERNEL=0`` keeps ring CP on the permute loop (the A/B switch behind the
#: ring-kernel measurement). Read once at import. Default ``1``: compiled, eligible ring
#: attention runs the NKI Library's ring kernel (``_ring_kernel_op``).
RING_KERNEL_ENV = "DIFFLET_NEURON_RING_KERNEL"
_RING_KERNEL = os.environ.get(RING_KERNEL_ENV, "1").strip() != "0"
#: The longest per-rank sequence the NKI Library documents its ring kernel as tested at.
_RING_KERNEL_MAX_SEQ = 16384
#: ``DIFFLET_NEURON_RING_CONSTMAX=0`` keeps eligible ring CP calls on the running-max ring kernel
#: (the A/B switch). Default ``1``: head dimension 128 runs the const-max ring kernel
#: (``_ring_const_max_op``), whose K-stationary attention lets the K/V permutes overlap compute.
RING_CONSTMAX_ENV = "DIFFLET_NEURON_RING_CONSTMAX"
_RING_CONSTMAX = os.environ.get(RING_CONSTMAX_ENV, "1").strip() != "0"
#: ``DIFFLET_NEURON_JOINT_RING_KERNEL=0`` keeps joint ring CP on the permute loop (the A/B switch).
#: Default ``1``: eligible ``joint_ring_attention`` calls run the joint const-max ring kernel.
JOINT_RING_KERNEL_ENV = "DIFFLET_NEURON_JOINT_RING_KERNEL"
_JOINT_RING_KERNEL = os.environ.get(JOINT_RING_KERNEL_ENV, "1").strip() != "0"
#: ``DIFFLET_NEURON_JOINT_RING_TEXT_SPLIT=0`` has every rank compute all text queries in the joint
#: ring kernel (the A/B switch). Default ``1``: each rank computes ceil(S_txt/cp) of them and the
#: text outputs are all-gathered over the CP group.
JOINT_RING_TEXT_SPLIT_ENV = "DIFFLET_NEURON_JOINT_RING_TEXT_SPLIT"
_JOINT_RING_TEXT_SPLIT = os.environ.get(JOINT_RING_TEXT_SPLIT_ENV, "1").strip() != "0"

#: Experiment E1 (evidence at 713d53d9), opt-in:
#: ``DIFFLET_EXP_ATTN_CONSTMAX=1`` sends unbounded, non-causal, bf16, d = 128 kernel calls that
#: carry their layer's score sigma (``constmax_sigma``, a [128, 1] tensor from the model's
#: committed table) to nkilib's experimental ``attention_const_max``; such calls with key-prefix
#: bounds (bound_min 0, bound_max equal along queries: Qwen-Image, HunyuanVideo, HV1.5) go to
#: Difflet's bounded fork of it (``attention_const_max_kernel.py``). Read once at import.
_CONSTMAX = os.environ.get("DIFFLET_EXP_ATTN_CONSTMAX") == "1"

#: ``DIFFLET_ATTN_QUANT=mxfp8`` sends the self/joint attention calls (non-causal, unmasked or
#: key-bounded, bf16, head dim 128, ``mx=True``) to ``difflet_neuron::mx_attention``: QK^T in
#: MXFP8, PV in bf16 (MXFP8 attention plan, Task 5; ruling BR28). Unset or ``off``: every graph is
#: unchanged. Read once at import, as above; trn3 and compile mode only (``application_base``).
ATTN_QUANT_ENV = "DIFFLET_ATTN_QUANT"
ATTN_QUANT = os.environ.get(ATTN_QUANT_ENV, "off").strip().lower()
if ATTN_QUANT not in ("off", "mxfp8"):
    raise ValueError(f"{ATTN_QUANT_ENV} must be unset, 'off' or 'mxfp8', got {ATTN_QUANT!r}")

# Kernel limits stated by torch-neuronx's SDPA gate, other than its 512-multiple lengths.
_KERNEL_DTYPES = (torch.bfloat16, torch.float16)
_KERNEL_MAX_HEAD_DIM = 128
_KERNEL_MAX_BATCH_HEADS = 512
#: The largest batch x seqlen_q x seqlen_k one bounded-kernel call takes; larger calls are split
#: over the heads (``_head_chunk``). Above some size the kernel's NKI compile fails outright with
#: "BIR emission failed: colz: body blob exceeds uint32 offset range": on trn2 (neuronx-cc 2.28,
#: nki 0.7) HunyuanVideo at 720x1280x129 failed at 6 heads x 119,168 x 119,168 padded joint
#: tokens (8.5e10) and again at 3 heads (4.26e10). nkilib tests attention_cte up to
#: 32 x 36864 x 36864 = 4.35e10 (``_MAX_BS_TIMES_SEQLEN_QK``), so that size is not a safe bound
#: for the bounded (range-select) path; half of it is used until the real bound is measured.
_BOUNDED_KERNEL_MAX_BH_SQ_SK = 16 * 36864 * 36864


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
    vector_exp: bool = False,
    mx: bool = True,
    constmax_sigma=None,
    **kwargs,
):
    """``vector_exp`` (flash kernel paths): False, the default, runs the softmax exp on the Scalar
    engine (attention_cte ``allow_vector_exp=False``); True keeps the kernel's own trn3 choice, the
    Vector engine's ``exponential``. On trn3 the Vector engine already evicts every score tile from
    PSUM (scale and row max; the bounded path adds a ``range_select``), so with the exp there too
    it bounds the kernel while the Scalar engine idles: the Scalar exp is 7.5-21% faster per call
    (Wan 81f 63.7 -> 52.0 ms, H3 768p bounded 66.4 -> 54.0 ms) with the same rel L2 against fp32
    attention; the outputs differ in their low bits, so a denoising trajectory lands on a nearby
    but different sample. No effect off
    trn3 B1: trn2 has no Vector exp and always uses the Scalar engine.

    ``mx=False`` keeps the call in bf16 under ``DIFFLET_ATTN_QUANT=mxfp8``: the cross-attention
    call sites pass it (plan Global Constraints: only self/joint attention is routed)."""
    if kwargs:
        raise NotImplementedError(
            "neuron attention does not support attention_cte-only kwargs: "
            + ", ".join(sorted(kwargs))
        )
    q = q if tp_q else q.transpose(-1, -2)
    k = k if tp_k else k.transpose(-1, -2)

    bounded = bound_min is not None or bound_max is not None
    if bounded:
        if bound_min is None or bound_max is None:
            raise ValueError("bound_min and bound_max must both be provided")
        if attention_mask is not None:
            raise ValueError("attention_mask and bounds are mutually exclusive")

    if mx and ATTN_QUANT == "mxfp8" and _routes_to_mx(q, k, v, causal, attention_mask):
        out = _mx_attention(q, k, v, bound_min, bound_max,
                            scale=1.0 if scale is None else float(scale))
        return out.transpose(-1, -2) if tp_out else out

    if bounded:
        if (_CONSTMAX and constmax_sigma is not None and not causal and _CONST_MAX_BOUNDED_OP is not None
                and _const_max_shape_ok(q, k, v, None) and q.shape[-2] >= 128):
            # The key-bounded fork: callers' key-prefix contract (bound_min 0, bound_max equal along
            # queries; the kernel reads query rows 0..127 of bound_max).
            out = _const_max_attention(q, k, v, constmax_sigma, scale=1.0 if scale is None else float(scale),
                                       bound_max=bound_max)
            return out.transpose(-1, -2) if tp_out else out
        if not causal and _can_use_flash_kernel(q, k, v, None) and _FLASH_BOUNDED_OP is not None:
            # attention_cte's sequence packing: per-query key range [bound_min, bound_max), read per
            # query tile inside the kernel (O(L) memory) instead of an L x L mask and masked SDPA.
            out = _flash_attention_bounded(q, k, v, bound_min, bound_max,
                                           scale=1.0 if scale is None else float(scale),
                                           vector_exp=vector_exp)
            return out.transpose(-1, -2) if tp_out else out
        key_idx = torch.arange(k.shape[-2], device=k.device).view(1, 1, -1)
        attention_mask = (key_idx >= bound_min.to(key_idx.device)) & (
            key_idx < bound_max.to(key_idx.device)
        )

    if attention_mask is not None and causal:
        attention_mask = _merge_causal_mask(q, k, attention_mask)
        causal = False

    scale = 1.0 if scale is None else float(scale)
    if (_CONSTMAX and constmax_sigma is not None and not causal and _CONST_MAX_OP is not None
            and _const_max_shape_ok(q, k, v, attention_mask)):
        out = _const_max_attention(q, k, v, constmax_sigma, scale=scale)
    elif _can_use_flash_kernel(q, k, v, attention_mask):
        out = _flash_attention(q, k, v, scale=scale, causal=causal, vector_exp=vector_exp)
    else:
        # Pass SDPA's default scale when it matches, keeping the call on its plain path.
        sdpa_scale = None if math.isclose(scale, 1.0 / math.sqrt(q.shape[-1])) else scale
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=causal, scale=sdpa_scale
        )
    return out.transpose(-1, -2) if tp_out else out


def _can_use_flash_kernel(q, k, v, attention_mask) -> bool:
    if _ATTENTION_PATH == "sdpa":
        return False
    if attention_mask is not None or not _on_neuron(q) or q.dtype not in _KERNEL_DTYPES:
        return False
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return False  # inference-only kernel call; autograd goes through SDPA
    if q.dim() not in (3, 4) or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        return False
    batch_heads = math.prod(q.shape[:-2])
    if q.shape[-1] > _KERNEL_MAX_HEAD_DIM or batch_heads > _KERNEL_MAX_BATCH_HEADS:
        return False
    return _flash_kernel() is not None


def _on_neuron(t) -> bool:
    return t.device.type == "neuron"


def _flash_attention(q, k, v, *, scale: float, causal: bool, vector_exp: bool = False):
    """Run the kernel on [B, H, S, d] (3D [B*H, S, d] gets a unit batch axis)."""
    squeeze = q.dim() == 3
    if squeeze:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    kernel = (not vector_exp and _flash_scalar_exp_kernel()) or _flash_kernel()
    out = kernel(q, k, v, is_causal=causal, scale=scale, training=False)
    return out.squeeze(0) if squeeze else out


def _load_flash_bounded_op(vector_exp: bool = True):
    """attention_cte with per-query key bounds (its "sequence packing"): ``(q, k, v, bound_min,
    bound_max, scale) -> out`` on ``[B*H, L, d]``, bounds int32 ``[B*H, L_q, 1]``, query ``i``
    attending to keys ``bound_min[i] <= j < bound_max[i]``. torch-neuronx's own SDPA wrapper does
    not forward bounds, so the kernel is called as its decomposition calls it (``tp_q``/``tp_k``
    layout, an LNC-wide grid), registered with ``nki_op`` like the logsumexp op."""
    if not vector_exp and not _EXP_KNOB:
        return None
    name = "flash_attention_bounded" if vector_exp else "flash_attention_bounded_scalar_exp"
    try:
        return getattr(torch.ops.difflet_neuron, name)  # already registered (module reload)
    except (AttributeError, RuntimeError):
        pass
    try:
        from torch_neuronx.neuron_dynamo_backend.decompositions import (
            get_logical_neuron_cores,
            wrapped_flash_fwd,
        )
        from torch_neuronx.nki_hop import nki_op
    except ImportError:
        return None

    # The vector_exp=True op passes no allow_vector_exp (the kernel's own default, its graph as
    # before the Scalar exp became the default); the Scalar-exp variant is a separate op.
    extra = {} if vector_exp else {"allow_vector_exp": False}

    @nki_op(f"difflet_neuron::{name}", mutates_args={})
    def flash_attention_bounded(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                bound_min: torch.Tensor, bound_max: torch.Tensor,
                                scale: float) -> torch.Tensor:
        grid = (int(get_logical_neuron_cores()),)
        return wrapped_flash_fwd[grid](q, k, v, tp_q=True, tp_k=True, scale=scale, causal_mask=False,
                                       cache_softmax=False, bound_min=bound_min, bound_max=bound_max,
                                       **extra)

    return flash_attention_bounded


def _routes_to_mx(q, k, v, causal: bool, attention_mask) -> bool:
    """The calls the MXFP8 fork takes: non-causal, unmasked (bounds aside), bf16, head dim 128,
    within the flash kernel's limits (``_can_use_flash_kernel``)."""
    return (_MX_ATTENTION_OP is not None and not causal and attention_mask is None
            and q.shape[-1] == 128 and all(t.dtype == torch.bfloat16 for t in (q, k, v))
            and _can_use_flash_kernel(q, k, v, None))


def _mx_op(q, k, v, bound_min, bound_max, scale: float):
    """The MXFP8 op in the routed variant, QK^T MXFP8 and PV bf16 (ruling BR28), called as
    ``_FLASH_BOUNDED_OP`` is."""
    return _MX_ATTENTION_OP(q, k, v, bound_min, bound_max, scale, False, True)


def _mx_attention(q, k, v, bound_min, bound_max, *, scale: float):
    """``[B, H, S, d]`` or ``[B*H, S, d]`` through the MXFP8 op. K is smoothed (minus its mean
    over the call's keys, ``smooth_keys``) before any padding. A bounded call then takes
    ``_flash_attention_bounded``'s q pre-scale, 128 padding, head chunks and one slice; an
    unbounded one is not padded: the fork takes partial 512-key and 128-query tiles itself
    (ruling BR3: one pad/slice round trip at most)."""
    shape = q.shape
    bh, s_q, d = math.prod(shape[:-2]), shape[-2], shape[-1]
    q3, k3, v3 = (t.reshape(bh, t.shape[-2], t.shape[-1]) for t in (q, k, v))
    k3 = smooth_keys(k3)
    record_call(s_q, k3.shape[1], bound_min is not None)
    if bound_min is not None:
        out = _flash_attention_bounded(q3, k3, v3, bound_min, bound_max, scale=scale, op=_mx_op)
    else:
        out = _mx_op(q3.contiguous(), k3.contiguous(), v3.contiguous(), None, None, scale)
    return out.reshape(*shape[:-1], v.shape[-1])


def _flash_attention_bounded(q, k, v, bound_min, bound_max, *, scale: float,
                             vector_exp: bool = True, op=None):
    """``[B, H, S, d]`` or ``[B*H, S, d]`` through the bounded kernel (``op``, default
    ``_FLASH_BOUNDED_OP``, its Scalar-exp variant with ``vector_exp=False``); bounds of
    ``[B*H, S_q, 1]`` (or anything that reshapes to it)."""
    shape = q.shape
    bh, s_q, d = math.prod(shape[:-2]), shape[-2], shape[-1]
    q3, k3, v3 = (t.reshape(bh, t.shape[-2], d) for t in (q, k, v))
    lo = bound_min.to(q.device, torch.int32).reshape(bh, s_q, 1)
    hi = bound_max.to(q.device, torch.int32).reshape(bh, s_q, 1)
    # The kernel's range_select (sequence packing) path requires scale == 1.0 (NCC_INKI016); as its
    # docstring prescribes, q is scaled first: softmax(s q k^T) = softmax((s q) k^T), with s q
    # computed in fp32 and rounded once to q's dtype.
    q3 = (q3.float() * scale).to(q3.dtype)
    # It also needs query and key lengths in multiples of 128 (other lengths fail to compile, a
    # DMACopy error; the unbounded kernel takes any length). Zero keys appended at index >= every
    # bound_max are excluded by the bounds; appended query rows get the one-key range [0, 1) (no
    # fully masked row) and are dropped.
    s_k = k3.shape[1]
    pad_q, pad_k = -s_q % 128, -s_k % 128
    if pad_k:
        k3, v3 = F.pad(k3, (0, 0, 0, pad_k)), F.pad(v3, (0, 0, 0, pad_k))
    if pad_q:
        q3 = F.pad(q3, (0, 0, 0, pad_q))
        lo = F.pad(lo, (0, 0, 0, pad_q))
        hi = F.pad(hi, (0, 0, 0, pad_q), value=1)
    if op is None:  # the MXFP8 path passes its own op
        op = (_FLASH_BOUNDED_OP if vector_exp
              else (_FLASH_BOUNDED_SCALAR_EXP_OP or _FLASH_BOUNDED_OP))
    chunk = _head_chunk(bh, q3.shape[1], k3.shape[1])
    if chunk == bh:  # one call on the whole tensors: the graph compiled before the split existed
        out = op(q3.contiguous(), k3.contiguous(), v3.contiguous(),
                 lo.contiguous(), hi.contiguous(), 1.0)
    else:
        out = torch.cat([op(q3[i:i + chunk].contiguous(),
                            k3[i:i + chunk].contiguous(),
                            v3[i:i + chunk].contiguous(),
                            lo[i:i + chunk].contiguous(),
                            hi[i:i + chunk].contiguous(), 1.0)
                         for i in range(0, bh, chunk)], dim=0)
    return out[:, :s_q].reshape(*shape[:-1], v.shape[-1])


def _head_chunk(bh: int, s_q: int, s_k: int) -> int:
    """Heads per bounded-kernel call: all ``bh`` when ``bh * s_q * s_k`` is within
    ``_BOUNDED_KERNEL_MAX_BH_SQ_SK`` (every shape the repo verified before 720p), else the largest
    divisor of ``bh`` that is (equal chunks: one kernel shape, traced once), at least 1."""
    fits = max(1, _BOUNDED_KERNEL_MAX_BH_SQ_SK // (s_q * s_k))
    return max(c for c in range(1, bh + 1) if bh % c == 0 and c <= fits)


def _load_flash_kernel():
    try:
        from torch_neuronx.python_ops.nki_kernels.scaled_dot_product_attention import (
            scaled_dot_product_attention_kernel,
        )
    except ImportError:
        return None
    return scaled_dot_product_attention_kernel


def _attention_cte_has_exp_knob() -> bool:
    """Whether the installed nkilib's attention_cte takes ``allow_vector_exp``. Older stacks do not;
    there every call keeps the kernel default (off trn3, the Scalar exp anyway)."""
    try:
        from nkilib.core.attention.attention_cte import attention_cte
    except ImportError:
        return False
    return "allow_vector_exp" in inspect.signature(attention_cte).parameters


def _flash_fwd(q, k, v, *, is_causal: bool, scale: float, cache_softmax: bool):
    """torch-neuronx's ``_compute_attention_nki`` on ``[B, H, L, d]`` with the softmax exp on the
    Scalar engine: its grid, reshapes and kernel arguments plus ``allow_vector_exp=False``, which its
    SDPA op cannot pass. With ``cache_softmax`` it returns ``(out, lse)``, the natural-log
    logsumexp ``[B, H, L]`` (fp32) from the kernel's tiled softmax stats, as torch-neuronx derives
    it. Kernel-lowering code: call it inside an ``nki_op`` only."""
    from torch_neuronx.neuron_dynamo_backend import decompositions as dec

    b, h, length, _ = q.shape
    h_kv, s, d_v = k.shape[1], k.shape[2], v.shape[-1]
    kernel = dec.wrapped_flash_fwd[(int(dec.get_logical_neuron_cores()),)]
    knob = {"allow_vector_exp": False} if _EXP_KNOB else {}
    result = kernel(q.reshape(b * h, length, q.shape[-1]), k.reshape(b * h_kv, s, k.shape[-1]),
                    v.reshape(b * h_kv, s, d_v), tp_q=True, tp_k=True, scale=scale,
                    causal_mask=is_causal, cache_softmax=cache_softmax, **knob)
    if not cache_softmax:
        return result.reshape(b, h, length, d_v)
    out, neg_max, out_sum_recip = result
    # The stats are tiled [B*H, 128, n_tile_q] (position t * 128 + p): back to sequence order.
    lse = -(neg_max + torch.log(out_sum_recip))
    lse = lse.to(torch.float32).reshape(b, h, *lse.shape[-2:]).transpose(-2, -1).reshape(b, h, -1)
    return out.reshape(b, h, length, d_v), lse


def _load_flash_scalar_exp_op():
    """The unbounded flash kernel with the Scalar exp (``_flash_fwd``), registered with ``nki_op``
    as torch-neuronx registers its SDPA kernel: ``(q, k, v, is_causal, scale) -> out`` on
    ``[B, H, L, d]``. ``None`` without torch-neuronx or the kernel knob."""
    if not _EXP_KNOB:
        return None
    try:
        return torch.ops.difflet_neuron.flash_attention_scalar_exp  # already registered (reload)
    except (AttributeError, RuntimeError):
        pass
    try:
        from torch_neuronx.neuron_dynamo_backend import decompositions  # noqa: F401
        from torch_neuronx.nki_hop import nki_op
    except ImportError:
        return None

    @nki_op("difflet_neuron::flash_attention_scalar_exp", mutates_args={})
    def flash_attention_scalar_exp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   is_causal: bool, scale: float) -> torch.Tensor:
        return _flash_fwd(q, k, v, is_causal=is_causal, scale=scale, cache_softmax=False)

    return flash_attention_scalar_exp


# Resolved once at import: no import or cache lookup inside traced (compiled) code. torch_neuronx
# before nkilib (with autoload off, this module can be its first importer): nkilib reads
# NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE at import, and torch_neuronx's import may set it.
_FLASH_KERNEL = _load_flash_kernel()
_EXP_KNOB = _attention_cte_has_exp_knob()
_FLASH_SCALAR_EXP_OP = _load_flash_scalar_exp_op()
_FLASH_BOUNDED_OP = _load_flash_bounded_op()
_FLASH_BOUNDED_SCALAR_EXP_OP = _load_flash_bounded_op(vector_exp=False)
_MX_ATTENTION_OP = load_op() if ATTN_QUANT == "mxfp8" else None  # unset / off: nothing loaded


def _load_flash_lse_op():
    """torch-neuronx's flash kernel returning ``(out, lse)``: unmasked attention on
    ``[B, H, L, d]`` and the natural-log logsumexp of its scaled scores, ``[B, H, L]`` fp32.

    The kernel's public wrapper hands the logsumexp back by copying into a caller buffer from an
    op registered as mutating nothing, which compiled graphs may drop; this op returns it instead,
    registered with ``nki_op`` like the wrapper and making the same kernel call
    (``_flash_fwd`` with ``cache_softmax=True``: torch-neuronx's ``training=True`` path, with the
    Scalar exp). That path requires the query length to be a multiple of 128 (kernel validation
    NCC_INKI016) where the plain forward does not, so the query is zero-padded and the padded rows
    are dropped; without a mask every query row is independent, so this is exact.
    """
    try:
        return torch.ops.difflet_neuron.flash_attention_lse  # already registered (module reload)
    except (AttributeError, RuntimeError):
        pass
    try:
        from torch_neuronx.neuron_dynamo_backend import decompositions  # noqa: F401
        from torch_neuronx.nki_hop import nki_op
    except ImportError:
        return None

    @nki_op("difflet_neuron::flash_attention_lse", mutates_args={})
    def flash_attention_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.shape[2]
        padded = -(-length // 128) * 128
        if padded != length:
            q = F.pad(q, (0, 0, 0, padded - length))
        out, lse = _flash_fwd(q, k, v, is_causal=False, scale=scale, cache_softmax=True)
        return out[:, :, :length].contiguous(), lse[..., :length].contiguous()

    return flash_attention_lse


_FLASH_LSE_OP = _load_flash_lse_op()


def _load_const_max_op():
    """nkilib's experimental ``attention_const_max`` (bf16, shipped unmodified, experiment E1):
    ``(q, k, v, scale, softmax_max) -> out`` on ``[B*H, S, d]``, q and k SEQ_MAJOR (the kernel
    DMA-transposes them), heads sharded over the LNC cores. ``softmax_max`` is the kernel's
    ``[128, 1]`` fp32 tensor form: a per-layer value as a graph input, so the blocks of one class
    keep one shared graph (a Python scalar would be baked in per layer). Registered with
    ``nki_op`` like the bounded op."""
    try:
        return torch.ops.difflet_neuron.attention_const_max  # already registered (module reload)
    except (AttributeError, RuntimeError):
        pass
    try:
        from nkilib.core.utils.common_types import AttentionLayout
        from nkilib.experimental.attention.attention_const_max import attention_const_max
        from torch_neuronx import wrap_nki
        from torch_neuronx.neuron_dynamo_backend.decompositions import get_logical_neuron_cores
        from torch_neuronx.nki_hop import nki_op
    except ImportError:
        return None
    kernel = wrap_nki(attention_const_max)

    @nki_op("difflet_neuron::attention_const_max", mutates_args={})
    def attention_const_max_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                               softmax_max: torch.Tensor) -> torch.Tensor:
        grid = (int(get_logical_neuron_cores()),)
        return kernel[grid](q, k, v, softmax_max=softmax_max, softmax_scale=scale,
                            q_layout=AttentionLayout.SEQ_MAJOR, k_layout=AttentionLayout.SEQ_MAJOR)

    return attention_const_max_op


def _load_const_max_bounded_op():
    """Difflet's fork of ``attention_const_max`` with a key-prefix bound
    (``ops_impl/attention_const_max_kernel.py``): ``(q, k, v, scale, softmax_max, hi) -> out``,
    E1's op plus ``hi``, int32 ``[B*H, S_q, 1]`` (attention_cte's ``bound_max``; keys
    ``[0, hi)`` attended, the same for every query)."""
    try:
        return torch.ops.difflet_neuron.attention_const_max_bounded  # already registered (module reload)
    except (AttributeError, RuntimeError):
        pass
    try:
        from nkilib.core.utils.common_types import AttentionLayout
        from torch_neuronx import wrap_nki
        from torch_neuronx.neuron_dynamo_backend.decompositions import get_logical_neuron_cores
        from torch_neuronx.nki_hop import nki_op

        from difflet.backends.neuron.ops_impl.attention_const_max_kernel import attention_const_max
    except ImportError:
        return None
    kernel = wrap_nki(attention_const_max)

    @nki_op("difflet_neuron::attention_const_max_bounded", mutates_args={})
    def attention_const_max_bounded_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                                       softmax_max: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        grid = (int(get_logical_neuron_cores()),)
        return kernel[grid](q, k, v, softmax_max=softmax_max, softmax_scale=scale, key_bound=hi,
                            q_layout=AttentionLayout.SEQ_MAJOR, k_layout=AttentionLayout.SEQ_MAJOR)

    return attention_const_max_bounded_op


_CONST_MAX_OP = _load_const_max_op() if _CONSTMAX else None
_CONST_MAX_BOUNDED_OP = _load_const_max_bounded_op() if _CONSTMAX else None


def _const_max_shape_ok(q, k, v, attention_mask) -> bool:
    return (_can_use_flash_kernel(q, k, v, attention_mask) and q.shape[-1] == 128
            and q.dtype == k.dtype == v.dtype == torch.bfloat16)


def _const_max_attention(q, k, v, sigma, *, scale: float, bound_max=None):
    """``[B, H, S, d]`` or ``[B*H, S, d]`` through the const-max op, softmax_max by the kernel
    docstring's rule at this call's key length: 0.88 sigma sqrt(2 ln Sk) (``sigma``: the layer's
    ``[128, 1]`` score sigma). With ``bound_max`` (anything that reshapes to ``[B*H, S_q, 1]``),
    the key-bounded op; Sk stays the padded key length (static, a slightly higher max)."""
    shape = q.shape
    softmax_max = sigma.float() * (0.88 * math.sqrt(2 * math.log(k.shape[-2])))
    q3, k3, v3 = (t.reshape(-1, t.shape[-2], t.shape[-1]).contiguous() for t in (q, k, v))
    if bound_max is None:
        out = _CONST_MAX_OP(q3, k3, v3, scale, softmax_max.contiguous())
    else:
        hi = bound_max.to(q.device, torch.int32).reshape(q3.shape[0], q3.shape[1], 1).contiguous()
        out = _CONST_MAX_BOUNDED_OP(q3, k3, v3, scale, softmax_max.contiguous(), hi)
    return out.reshape(*shape[:-1], v.shape[-1])


def _attention_with_lse(q, k, v, *, scale: float):
    """Unmasked attention on ``[B, H, L, d]`` and its logsumexp ``[B, H, L]`` (fp32): the flash
    kernel where it applies, else fp32 math (CPU, fp32 inputs)."""
    if _FLASH_LSE_OP is not None and _can_use_flash_kernel(q, k, v, None):
        return _FLASH_LSE_OP(q, k, v, scale)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)
    return out, lse


def _flash_kernel():
    return _FLASH_KERNEL


def _scalar_exp_sdpa(q, k, v, *, is_causal, scale, training):
    """``_FLASH_SCALAR_EXP_OP`` behind torch-neuronx's SDPA kernel signature (inference only)."""
    assert not training
    return _FLASH_SCALAR_EXP_OP(q, k, v, is_causal, scale)


def _flash_scalar_exp_kernel():
    return None if _FLASH_SCALAR_EXP_OP is None else _scalar_exp_sdpa


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
        mx=False,
        **kwargs,
    )


def _merge_causal_mask(q, k, attention_mask):
    q_len, k_len = q.shape[-2], k.shape[-2]
    causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
    if attention_mask.dtype == torch.bool:
        return attention_mask & causal_mask
    causal_bias = torch.zeros((q_len, k_len), dtype=q.dtype, device=q.device)
    causal_bias = causal_bias.masked_fill(~causal_mask, torch.finfo(q.dtype).min)
    return attention_mask.to(device=q.device, dtype=q.dtype) + causal_bias


def _dense_bhsd(q, k, v, *, scale: float):
    """Unmasked attention on ``[B, H, S, d]`` in the standard layout, through ``attention``'s
    ``attention_cte`` convention (heads folded into the batch, ``tp_q``/``tp_k`` set), as Wan's
    own attention path calls it. Its callers, the CP variants, stay bf16 (``mx=False``)."""
    b, h, s_q, d = q.shape
    s_k = k.shape[2]
    out = attention(q.reshape(b * h, s_q, d), k.reshape(b * h, s_k, d), v.reshape(b * h, s_k, d),
                    scale=scale, causal=False, tp_q=True, tp_k=True, tp_out=False, mx=False)
    return out.reshape(b, h, s_q, d)


def ring_attention(q, k, v, *, scale: float, causal: bool = False, token_major: bool = False):
    """Ring context-parallel self-attention over a sequence sharded on the CP group.

    ``q, k, v``: ``[B, H, S/cp, d]`` to ``[B, H, S/cp, d]``; with ``token_major``, ``q, k, v`` are
    ``[B, S/cp, H, d]`` (the projections' own layout, which the const-max ring kernel takes as is)
    and the output is still ``[B, H, S/cp, d]``. In ``cp`` rounds each rank attends
    its query shard to the K/V block it holds, getting ``(o_r, l_r)`` (output, logsumexp), and
    passes the block to the next rank. Partial results merge in fp32 by their logsumexp
    (``_merge_partials``), which is exactly the softmax over all ``S`` keys. Non-causal only. Any shard length: the flash kernel's
    logsumexp path needs query lengths in multiples of 128, which ``_attention_with_lse`` pads to.
    """
    if causal:
        raise NotImplementedError("ring_attention supports non-causal attention only")
    token_major_qkv = (q, k, v) if token_major else None
    if token_major:  # [B, H, S/cp, d] views for every other path: no copy
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    from difflet.backends.neuron.ops_impl import collectives
    from difflet.backends.neuron.ops_impl.parallel_mesh import get_axis_rank, get_axis_size

    cp = get_axis_size("cp")
    if cp == 1:
        return _dense_bhsd(q, k, v, scale=scale)
    if _RING_CONSTMAX_OP is not None and _ring_kernel_eligible(q, k, v) and q.shape[-1] == 128:
        # [B, S/cp, H, d]: its layout, as the caller passed it, else from the head-major tensors.
        tm = token_major_qkv or tuple(x.transpose(1, 2) for x in (q, k, v))
        out = _RING_CONSTMAX_OP(*(x.contiguous() for x in tm), scale)
        return out.reshape(q.shape)
    if _RING_KERNEL_OP is not None and _ring_kernel_eligible(q, k, v):
        return _RING_KERNEL_OP(q, k, v, scale).reshape(q.shape)
    out, _ = _ring_partials(q, k, v, scale=scale, cp=cp,
                            group=collectives.get_cp_group(), rank=get_axis_rank("cp"))
    return out.to(q.dtype)


def _ring_kernel_eligible(q, k, v) -> bool:
    """Whether the NKI Library ring kernel takes this call: compiled (the kernel issues its own
    collective permutes), plain multi-head self-attention ``[B, H, S/cp, d]`` with equal Q, K and
    V shapes, a kernel dtype, ``d <= 128`` and a per-rank length the kernel is documented at."""
    return (_RING_KERNEL and torch.compiler.is_compiling() and q.dim() == 4
            and q.shape == k.shape == v.shape and q.dtype in _KERNEL_DTYPES
            and q.shape[-1] <= _KERNEL_MAX_HEAD_DIM and q.shape[2] <= _RING_KERNEL_MAX_SEQ)


_RING_KERNEL_OPS: dict = {}
#: The ring kernel op for the current mesh's CP rings, set by ``prepare_ring_kernel``.
_RING_KERNEL_OP = None
_RING_CONSTMAX_OPS: dict = {}
#: The const-max ring kernel op for the current mesh's CP rings, set by ``prepare_ring_kernel``.
_RING_CONSTMAX_OP = None
_JOINT_RING_OPS: dict = {}
#: The joint const-max ring kernel op for the mesh's CP rings, set by ``prepare_ring_kernel``.
_JOINT_RING_OP = None


def prepare_ring_kernel(spec) -> None:
    """Register the ring kernel op for ``spec``'s CP rings, eagerly: Dynamo cannot trace the op's
    registration, so it happens at mesh initialization, before any compilation, and
    ``ring_attention`` only reads the result. ``None`` without CP, when switched off, or when
    the kernel is unavailable."""
    global _RING_KERNEL_OP, _RING_CONSTMAX_OP, _JOINT_RING_OP
    if not _RING_KERNEL or spec.axis_size("cp") == 1:
        _RING_KERNEL_OP = _RING_CONSTMAX_OP = _JOINT_RING_OP = None
        return
    rings = tuple(tuple(g) for g in spec.axis_groups("cp"))
    _RING_KERNEL_OP = _ring_kernel_op(rings)
    _RING_CONSTMAX_OP = _ring_const_max_op(rings) if _RING_CONSTMAX else None
    _JOINT_RING_OP = _joint_ring_op(rings) if _JOINT_RING_KERNEL else None


def _ring_kernel_op(rings: tuple):
    """``(q, k, v, scale) -> out`` on ``[B, H, S/cp, d]``: the NKI Library's
    ``ring_attention_spmd_fwd``, which overlaps each step's K/V ``collective_permute`` with the
    previous step's softmax correction, so the whole ring is one kernel. ``rings`` are the CP
    groups in global ranks (its ``replica_groups``); a custom-op schema cannot carry them, so one
    op is registered per ring set. ``None`` when the kernel or ``nki_op`` is unavailable."""
    if rings in _RING_KERNEL_OPS:
        return _RING_KERNEL_OPS[rings]
    import hashlib

    name = "ring_attention_" + hashlib.sha1(repr(rings).encode()).hexdigest()[:12]
    try:
        op = getattr(torch.ops.difflet_neuron, name)  # already registered (module reload)
    except (AttributeError, RuntimeError):
        try:
            # Difflet's fork: its exp on the Scalar engine on Trainium3 (ring_attention_fwd_kernel.py).
            from difflet.backends.neuron.ops_impl.ring_attention_fwd_kernel import ring_attention_spmd_fwd
            from torch_neuronx.neuron_dynamo_backend.decompositions import get_logical_neuron_cores
            from torch_neuronx.nki_hop import nki_op
        except ImportError:
            _RING_KERNEL_OPS[rings] = None
            return None
        num_workers = len(rings[0])

        @nki_op(f"difflet_neuron::{name}", mutates_args={})
        def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
            lnc = int(get_logical_neuron_cores())  # this kernel takes the LNC, not a grid tuple
            return ring_attention_spmd_fwd[lnc](q, k, v, replica_groups=rings,
                                                num_workers=num_workers, softmax_scale=scale,
                                                tp_q=True, tp_k=True)
    _RING_KERNEL_OPS[rings] = op
    return op


def _ring_const_max_op(rings: tuple):
    """``(q, k, v, scale) -> out``, q/k/v token-major ``[B, S/cp, H, 128]``, out
    ``[B, H, S/cp, 128]``: vLLM-Omni Neuron's const-max ring kernel
    (``ring_const_max_fwd_kernel.py``). Its softmax max is a
    per-row Cauchy-Schwarz bound computed in the kernel, so the ring merge is a plain sum, and its
    K-stationary attention has no ``dma_transpose``, which cannot run beside a collective: the K/V
    permutes overlap the attention itself. One op per ring set, as ``_ring_kernel_op``; ``None``
    when the kernel or ``nki_op`` is unavailable."""
    if rings in _RING_CONSTMAX_OPS:
        return _RING_CONSTMAX_OPS[rings]
    import hashlib

    name = "ring_const_max_" + hashlib.sha1(repr(rings).encode()).hexdigest()[:12]
    try:
        op = getattr(torch.ops.difflet_neuron, name)  # already registered (module reload)
    except (AttributeError, RuntimeError):
        try:
            from difflet.backends.neuron.ops_impl.ring_const_max_fwd_kernel import (
                ring_attention_const_max_fwd,
            )
            from torch_neuronx.neuron_dynamo_backend.decompositions import get_logical_neuron_cores
            from torch_neuronx.nki_hop import nki_op
        except ImportError:
            _RING_CONSTMAX_OPS[rings] = None
            return None
        num_workers = len(rings[0])

        @nki_op(f"difflet_neuron::{name}", mutates_args={})
        def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
            lnc = int(get_logical_neuron_cores())
            return ring_attention_const_max_fwd[lnc](q, k, v, replica_groups=rings,
                                                     num_workers=num_workers, softmax_scale=scale)
    _RING_CONSTMAX_OPS[rings] = op
    return op


def _joint_ring_op(rings: tuple):
    """``(q, k, v, k_txt, v_txt, key_mask, scale) -> out``, all token-major ``[B, S, H, 128]``:
    q has this rank's image rows then all text rows, k/v the image block, k_txt/v_txt the
    replicated text block (masked rows zeroed), key_mask ``[B*H, S_txt]`` 1/0; out
    ``[B, H, S_q, 128]`` in q's row order. The joint const-max ring kernel
    (``joint_ring_const_max_fwd_kernel.py``); one op per ring set, as ``_ring_kernel_op``."""
    if rings in _JOINT_RING_OPS:
        return _JOINT_RING_OPS[rings]
    import hashlib

    name = "joint_ring_const_max_" + hashlib.sha1(repr(rings).encode()).hexdigest()[:12]
    try:
        op = getattr(torch.ops.difflet_neuron, name)  # already registered (module reload)
    except (AttributeError, RuntimeError):
        try:
            from difflet.backends.neuron.ops_impl.joint_ring_const_max_fwd_kernel import (
                joint_ring_attention_const_max_fwd,
            )
            from torch_neuronx.neuron_dynamo_backend.decompositions import get_logical_neuron_cores
            from torch_neuronx.nki_hop import nki_op
        except ImportError:
            _JOINT_RING_OPS[rings] = None
            return None
        num_workers = len(rings[0])

        @nki_op(f"difflet_neuron::{name}", mutates_args={})
        def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_txt: torch.Tensor,
               v_txt: torch.Tensor, key_mask: torch.Tensor, scale: float) -> torch.Tensor:
            lnc = int(get_logical_neuron_cores())
            return joint_ring_attention_const_max_fwd[lnc](
                q, k, v, k_txt, v_txt, key_mask, softmax_scale=scale, replica_groups=rings,
                num_workers=num_workers)
    _JOINT_RING_OPS[rings] = op
    return op


def _joint_ring_kernel_eligible(q, image_k, image_v, text_k, text_v) -> bool:
    """Compiled, head dim 128, a kernel dtype, equal heads, ``q`` holding exactly the image and
    text rows, and an image block within the length the ring kernels are measured at."""
    return (torch.compiler.is_compiling() and q.dim() == 4 and q.shape[-1] == 128
            and q.dtype in _KERNEL_DTYPES and image_k.shape == image_v.shape
            and text_k.shape == text_v.shape and q.shape[1] == image_k.shape[1] == text_k.shape[1]
            and q.shape[2] == image_k.shape[2] + text_k.shape[2]
            and image_k.shape[2] <= _RING_KERNEL_MAX_SEQ)


def _text_split_rows(s_txt: int, cp: int) -> int:
    """Text query rows per rank under the text split: ceil(S_txt / cp), the last share padded."""
    return -(-s_txt // cp)


def _joint_ring_kernel(op, q, image_k, image_v, text_k, text_v, *, scale: float,
                       text_valid_len=None, text_first: bool = False, text_split=None):
    """Adapt a head-major ``joint_ring_attention`` call to the joint kernel: image rows first,
    masked text K/V rows zeroed (out of the softmax bound and of Σ P·V), a per-key 1/0 mask
    (out of Σ P·1), token-major tensors; the output comes back in ``q``'s row order.

    ``text_split`` = ``(cp, rank, gather)``: the kernel takes only this rank's share of the text
    queries, rows ``[rank·T, (rank+1)·T)`` with ``T = ceil(S_txt/cp)`` (zero-padded at the end),
    and ``gather`` all-gathers the text outputs along the sequence over the CP group. Every rank
    computing all text rows is redundant: each one sees every key."""
    b, h, s_q, _ = q.shape
    s_txt = text_k.shape[2]
    if text_first:
        q = torch.cat([q[:, :, s_txt:], q[:, :, :s_txt]], dim=2)
    s_img = s_q - s_txt
    if text_split is not None:
        cp, rank, gather = text_split
        rows = _text_split_rows(s_txt, cp)
        q_txt = F.pad(q[:, :, s_img:], (0, 0, 0, rows * cp - s_txt))
        q = torch.cat([q[:, :, :s_img], q_txt[:, :, rank * rows:(rank + 1) * rows]], dim=2)
    if text_valid_len is None:
        key_mask = torch.ones(b * h, s_txt, dtype=q.dtype, device=q.device)
    else:
        valid = (torch.arange(s_txt, device=q.device)[None, :]
                 < text_valid_len.to(torch.int32)[:, None]).to(q.dtype)  # [B, S_txt]
        text_k = text_k * valid[:, None, :, None]
        text_v = text_v * valid[:, None, :, None]
        key_mask = valid[:, None, :].expand(b, h, s_txt).reshape(b * h, s_txt)
    token_major = lambda x: x.transpose(1, 2).contiguous()  # noqa: E731
    out = op(token_major(q), token_major(image_k), token_major(image_v), token_major(text_k),
             token_major(text_v), key_mask.contiguous(), scale)
    if text_split is not None:  # this rank's text rows → every rank's, unpadded
        text_out = gather(out[:, :, s_img:].contiguous())[:, :, :s_txt]
        out = torch.cat([out[:, :, :s_img], text_out], dim=2)
    if text_first:
        out = torch.cat([out[:, :, s_q - s_txt:], out[:, :, :s_q - s_txt]], dim=2)
    return out


def _ring_partials(q, k, v, *, scale: float, cp: int, group, rank: int, out=None, lse=None):
    """The ring loop shared by ``ring_attention`` and ``joint_ring_attention``.

    Attends ``q`` to the K/V block this rank holds, passes the block to the next rank, and
    repeats ``cp`` times, so every rank's block is seen once. Each partial merges into the
    running ``(out, lse)`` (fp32 output, logsumexp), which a caller may seed with a partial of
    its own (the joint variant's text partial); returns the merged pair, output still fp32.
    """
    from difflet.backends.neuron.ops_impl import collectives

    kv = torch.stack([k, v])  # one hand-off per round for both
    for step in range(cp):
        o_r, l_r = _attention_with_lse(q, kv[0], kv[1], scale=scale)
        if out is None:
            out, lse = o_r.float(), l_r
        else:
            out, lse = _merge_partials(out, lse, o_r.float(), l_r)
        if step < cp - 1:
            kv = collectives.ring_shift(kv, group, cp, rank)
    return out, lse


def joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale: float,
                         causal: bool = False, text_valid_len=None, text_first: bool = False):
    """Ring context-parallel joint (text + image) attention: image sharded, text replicated.

    ``q``: this rank's joint queries ``[B, H, S_txt + S_img/cp, d]`` (the caller's order; Qwen
    puts text first). ``image_k, image_v``: ``[B, H, S_img/cp, d]``, this rank's image block.
    ``text_k, text_v``: ``[B, H, S_txt, d]``, identical on every rank. Returns
    ``[B, H, q.shape[2], d]`` in ``q``'s row order.

    The text K/V give one local partial, computed once and never passed on: every rank already
    holds them, and rotating them would count them ``cp`` times. That partial seeds the ring
    over the image blocks (``_ring_partials``, as in ``ring_attention``), and the ``cp + 1``
    partials merge by their logsumexp into exactly the softmax over all ``S_txt + S_img`` keys.
    Each partial attends the long joint query to a shorter key block (``L_q != L_k``), which
    ``_attention_with_lse`` takes on both the flash kernel and the fp32 path. Non-causal only:
    a causal mask across ring-rotated image blocks would need per-hop position offsets, and no
    joint caller is causal. At ``cp == 1`` this is plain attention over ``[text ‖ image]``.

    ``text_first``: ``q``'s rows are ``[text ‖ image]`` (Qwen-Image), not ``[image ‖ text]``;
    only the joint ring kernel, which takes image rows first, needs to know.

    Compiled calls with head dim 128 run the joint const-max ring kernel (``_joint_ring_kernel``):
    one kernel per attention, the text block attended once before the ring, the K/V permutes
    overlapped with the attention. Other calls run the loop below.

    ``text_valid_len`` (``[B]``, optional): text keys at or beyond it are padding (the caller
    puts valid text tokens first, as HunyuanVideo 1.5 does). They are masked in the text partial
    only, computed densely (the text block is short) with a key-index mask against the count, so
    no shape depends on it; image keys are all valid.
    """
    if causal:
        raise NotImplementedError("joint_ring_attention supports non-causal attention only")
    from difflet.backends.neuron.ops_impl import collectives
    from difflet.backends.neuron.ops_impl.parallel_mesh import get_axis_rank, get_axis_size

    cp = get_axis_size("cp")
    if (cp > 1 and _JOINT_RING_OP is not None
            and _joint_ring_kernel_eligible(q, image_k, image_v, text_k, text_v)):
        split = None
        if _JOINT_RING_TEXT_SPLIT:
            group = collectives.get_cp_group()

            def gather(x):
                return collectives.gather_from_tensor_model_parallel_region_with_dim(
                    x, gather_dim=2, process_group=group)
            split = (cp, get_axis_rank("cp"), gather)
        return _joint_ring_kernel(_JOINT_RING_OP, q, image_k, image_v, text_k, text_v, scale=scale,
                                  text_valid_len=text_valid_len, text_first=text_first,
                                  text_split=split)
    if cp == 1 and text_valid_len is None:
        return _dense_bhsd(q, torch.cat([text_k, image_k], dim=2),
                           torch.cat([text_v, image_v], dim=2), scale=scale)
    if text_valid_len is None:
        o_txt, l_txt = _attention_with_lse(q, text_k, text_v, scale=scale)
    else:
        o_txt, l_txt = _masked_attention_with_lse(q, text_k, text_v, scale=scale, key_valid_len=text_valid_len)
    if cp == 1:
        o_img, l_img = _attention_with_lse(q, image_k, image_v, scale=scale)
        out, _ = _merge_partials(o_txt.float(), l_txt, o_img.float(), l_img)
        return out.to(q.dtype)
    out, _ = _ring_partials(q, image_k, image_v, scale=scale, cp=cp,
                            group=collectives.get_cp_group(), rank=get_axis_rank("cp"),
                            out=o_txt.float(), lse=l_txt)
    return out.to(q.dtype)


def _masked_attention_with_lse(q, k, v, *, scale: float, key_valid_len):
    """Dense attention on ``[B, H, L, d]`` with keys at or beyond ``key_valid_len[b]`` masked, and its
    logsumexp ``[B, H, L]`` (fp32), max-subtracted (not ``torch.logsumexp``: the compiled
    ``logaddexp`` bug). For short key blocks (a padded text stream).

    A row with no valid key (``key_valid_len[b] == 0``) is the empty partial: output 0 and
    logsumexp -inf, which ``_merge_partials`` weighs 0. The max is floored at fp32's lowest
    finite value so ``exp(-inf - m)`` is 0 rather than ``exp(-inf - -inf)`` = NaN, and the sum at
    a tiny positive value so ``0 / 0`` cannot occur; with a valid key the max is finite and the
    sum at least 1, so both floors leave the result unchanged."""
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    keys = torch.arange(k.shape[-2], device=k.device).view(1, 1, 1, -1)
    valid = keys < key_valid_len.to(k.device).view(-1, 1, 1, 1)
    scores = scores.masked_fill(~valid, float("-inf"))
    m = scores.amax(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).min)
    p = torch.exp(scores - m)
    total = p.sum(dim=-1, keepdim=True)
    out = torch.matmul(p / total.clamp_min(torch.finfo(torch.float32).tiny), v.float()).to(q.dtype)
    return out, (m + torch.log(total)).squeeze(-1)


def _merge_partials(out, lse, o_r, l_r):
    """Merge two partial attentions by their logsumexp, max-subtracted so no exponent exceeds 0.

    With ``m = max(l, l_r)``, ``w = exp(l - m)`` and ``w_r = exp(l_r - m)`` (both in [0, 1]):
    ``o = (w o + w_r o_r) / (w + w_r)`` and ``l = m + log(w + w_r)``. Not ``torch.logaddexp``:
    compiled on the neuron backend it returns garbage from about 80 and inf from about 90 (eager
    PyTorch is stable), and attention logsumexps reach that range.
    """
    m = torch.maximum(lse, l_r)
    w, w_r = torch.exp(lse - m), torch.exp(l_r - m)
    total = w + w_r
    merged = (out * w.unsqueeze(-1) + o_r * w_r.unsqueeze(-1)) / total.unsqueeze(-1)
    return merged, m + torch.log(total)


def ulysses_attention(q, k, v, *, scale: float, causal: bool = False):
    """Ulysses context-parallel self-attention over a sequence sharded on the CP group.

    ``q, k, v``: ``[B, H, S/cp, d]`` (``H`` this TP rank's heads) to ``[B, H, S/cp, d]``. One
    all-to-all trades the sequence shard for a head shard, ``[B, H/cp, S, d]``, so ordinary
    attention sees the whole sequence (the flash kernel where it applies); a second restores
    the caller's layout. Exact: one dense softmax over the full sequence. Non-causal only: the
    swaps rebuild every rank's tokens but a caller sharding a joint sequence would not get
    global order back, so a causal mask could land on the wrong positions.
    """
    if causal:
        raise NotImplementedError("ulysses_attention supports non-causal attention only")
    from difflet.backends.neuron.ops_impl import collectives
    from difflet.backends.neuron.ops_impl.parallel_mesh import get_axis_size

    cp = get_axis_size("cp")
    if cp == 1:
        return _dense_bhsd(q, k, v, scale=scale)
    _check_ulysses_heads(q.shape[1], cp)
    group = collectives.get_cp_group()
    q, k, v = (collectives.seq_to_heads(t, group, cp) for t in (q, k, v))
    out = _dense_bhsd(q, k, v, scale=scale)
    return collectives.heads_to_seq(out, group, cp)


def _check_ulysses_heads(heads: int, cp: int) -> None:
    if heads % cp:
        raise ValueError(
            f"cp_mode='ulysses' needs this rank's head count ({heads}) divisible by "
            f"cp_degree ({cp}): num_attention_heads must be divisible by tp_degree * cp_degree"
        )


def joint_ulysses_attention(q_img, q_txt, image_k, image_v, text_k, text_v, *, scale: float,
                            causal: bool = False, key_valid_len=None):
    """Ulysses context-parallel joint (text + image) attention: image sharded, text replicated.

    ``q_img, image_k, image_v``: ``[B, H, S_img/cp, d]``, this rank's image block (``H`` this
    TP rank's heads). ``q_txt, text_k, text_v``: ``[B, H, S_txt, d]``, identical on every rank.
    Returns ``(img_out [B, H, S_img/cp, d], txt_out [B, H, S_txt, d])``, each stream in the
    sharding it arrived with, so each caller re-joins them in its own order (Qwen
    ``[txt ‖ img]``, HunyuanVideo ``[img ‖ txt]``).

    The image stream takes the Ulysses swap (``seq_to_heads``) to ``[B, H/cp, S_img, d]``. The
    text stream is replicated, so no exchange is needed to put it on the same heads: this rank
    narrows it to its own head block, ``H/cp`` heads from ``cp_rank * H/cp`` (the block
    ``seq_to_heads`` delivers here; the rank is a Python constant in this one-process-per-rank
    backend, so the slice is static). One dense attention over ``[image ‖ text]`` on those heads
    at full sequence length is then the exact joint result. On the way out the image rows take
    ``heads_to_seq`` back to the sequence shard, and the text rows are all-gathered over heads,
    restoring the replication callers expect.

    ``key_valid_len`` (int32 ``[B]`` or None): valid keys per batch row in the joint
    ``[image_full ‖ text]`` order, for a right-padded text stream (HunyuanVideo); keys at index
    ``>= key_valid_len[b]`` are masked for every query, through ``attention``'s bounds (masked,
    so SDPA rather than the flash kernel). Non-causal only, as ``ulysses_attention``.
    """
    if causal:
        raise NotImplementedError("joint_ulysses_attention supports non-causal attention only")
    from difflet.backends.neuron.ops_impl import collectives
    from difflet.backends.neuron.ops_impl.parallel_mesh import get_axis_rank, get_axis_size

    cp = get_axis_size("cp")
    if cp > 1:
        _check_ulysses_heads(q_img.shape[1], cp)
        group, heads = collectives.get_cp_group(), q_img.shape[1] // cp
        start = get_axis_rank("cp") * heads
        q_img, image_k, image_v = (collectives.seq_to_heads(t, group, cp)
                                   for t in (q_img, image_k, image_v))
        q_txt, text_k, text_v = (t.narrow(1, start, heads) for t in (q_txt, text_k, text_v))

    # Image first, the order key_valid_len counts in; each stream is returned separately.
    q = torch.cat([q_img, q_txt], dim=2)
    k = torch.cat([image_k, text_k], dim=2)
    v = torch.cat([image_v, text_v], dim=2)
    if key_valid_len is None:
        out = _dense_bhsd(q, k, v, scale=scale)
    else:
        b, h, s, d = q.shape
        count = key_valid_len.to(torch.int32).reshape(b, 1, 1, 1)
        bound_max = count.expand(b, h, s, 1).reshape(b * h, s, 1)
        out = attention(q.reshape(b * h, s, d), k.reshape(b * h, s, d), v.reshape(b * h, s, d),
                        scale=scale, causal=False, bound_min=torch.zeros_like(bound_max),
                        bound_max=bound_max, tp_q=True, tp_k=True, tp_out=False, mx=False)
        out = out.reshape(b, h, s, d)
    s_img = q_img.shape[2]
    img_out, txt_out = out[:, :, :s_img], out[:, :, s_img:]
    if cp > 1:
        img_out = collectives.heads_to_seq(img_out.contiguous(), group, cp)
        txt_out = collectives.gather_from_tensor_model_parallel_region_with_dim(
            txt_out.contiguous(), 1, group)
    return img_out, txt_out

__all__ = [
    "attention",
    "cross_attention",
    "ring_attention",
    "joint_ring_attention",
    "ulysses_attention",
    "joint_ulysses_attention",
]
