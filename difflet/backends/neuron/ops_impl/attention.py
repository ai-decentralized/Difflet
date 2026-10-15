"""Attention on the neuron device: the NKI flash kernel, with SDPA as the fallback.

Unmasked inference attention calls torch-neuronx's NKI flash-attention kernel
directly. Through ``F.scaled_dot_product_attention`` the kernel is reached only
when both sequence lengths are multiples of 512; the kernel itself takes other
lengths (Wan: 4,680 tokens), so the direct call avoids the decomposed path.
The kernel computes in bf16 internally, so only bf16/fp16 inputs take it; fp32 keeps
fp32 precision through SDPA. Masked or autograd calls, and shapes outside the
kernel's limits, also use SDPA.

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
The joint (text + image) variants are not implemented.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

#: ``DIFFLET_NEURON_ATTENTION=sdpa`` sends every call through ``F.scaled_dot_product_attention``
#: (never the direct kernel call): the A/B switch behind the SDPA-gate measurement. Read once at
#: import, so compiled regions see a constant. Default ``auto``: the direct kernel when eligible.
ATTENTION_PATH_ENV = "DIFFLET_NEURON_ATTENTION"
_ATTENTION_PATH = os.environ.get(ATTENTION_PATH_ENV, "auto").strip().lower()
if _ATTENTION_PATH not in ("auto", "sdpa"):
    raise ValueError(f"{ATTENTION_PATH_ENV} must be 'auto' or 'sdpa', got {_ATTENTION_PATH!r}")

# Kernel limits stated by torch-neuronx's SDPA gate, other than its 512-multiple lengths.
_KERNEL_DTYPES = (torch.bfloat16, torch.float16)
_KERNEL_MAX_HEAD_DIM = 128
_KERNEL_MAX_BATCH_HEADS = 512


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
    if kwargs:
        raise NotImplementedError(
            "neuron attention does not support attention_cte-only kwargs: "
            + ", ".join(sorted(kwargs))
        )
    q = q if tp_q else q.transpose(-1, -2)
    k = k if tp_k else k.transpose(-1, -2)

    if bound_min is not None or bound_max is not None:
        if bound_min is None or bound_max is None:
            raise ValueError("bound_min and bound_max must both be provided")
        if attention_mask is not None:
            raise ValueError("attention_mask and bounds are mutually exclusive")
        key_idx = torch.arange(k.shape[-2], device=k.device).view(1, 1, -1)
        attention_mask = (key_idx >= bound_min.to(key_idx.device)) & (
            key_idx < bound_max.to(key_idx.device)
        )

    if attention_mask is not None and causal:
        attention_mask = _merge_causal_mask(q, k, attention_mask)
        causal = False

    scale = 1.0 if scale is None else float(scale)
    if _can_use_flash_kernel(q, k, v, attention_mask):
        out = _flash_attention(q, k, v, scale=scale, causal=causal)
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


def _flash_attention(q, k, v, *, scale: float, causal: bool):
    """Run the kernel on [B, H, S, d] (3D [B*H, S, d] gets a unit batch axis)."""
    squeeze = q.dim() == 3
    if squeeze:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    out = _flash_kernel()(q, k, v, is_causal=causal, scale=scale, training=False)
    return out.squeeze(0) if squeeze else out


def _load_flash_kernel():
    try:
        from torch_neuronx.python_ops.nki_kernels.scaled_dot_product_attention import (
            scaled_dot_product_attention_kernel,
        )
    except ImportError:
        return None
    return scaled_dot_product_attention_kernel


# Resolved once at import: no import or cache lookup inside traced (compiled) code.
_FLASH_KERNEL = _load_flash_kernel()


def _load_flash_lse_op():
    """torch-neuronx's flash kernel returning ``(out, lse)``: unmasked attention on
    ``[B, H, L, d]`` and the natural-log logsumexp of its scaled scores, ``[B, H, L]`` fp32.

    The kernel's public wrapper hands the logsumexp back by copying into a caller buffer from an
    op registered as mutating nothing, which compiled graphs may drop; this op returns it instead,
    registered with ``nki_op`` like the wrapper and calling the same internal function
    (``_compute_attention_nki`` with ``training=True``, i.e. ``cache_softmax=True``). That path
    requires the query length to be a multiple of 128 (kernel validation NCC_INKI016) where the
    plain forward does not, so the query is zero-padded and the padded rows are dropped; without
    a mask every query row is independent, so this is exact.
    """
    try:
        return torch.ops.difflet_neuron.flash_attention_lse  # already registered (module reload)
    except (AttributeError, RuntimeError):
        pass
    try:
        from torch_neuronx.neuron_dynamo_backend.decompositions import _compute_attention_nki
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
        out, lse = _compute_attention_nki(q, k, v, False, scale, True)
        return out[:, :, :length].contiguous(), lse[..., :length].contiguous()

    return flash_attention_lse


_FLASH_LSE_OP = _load_flash_lse_op()


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


def _merge_causal_mask(q, k, attention_mask):
    q_len, k_len = q.shape[-2], k.shape[-2]
    causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
    if attention_mask.dtype == torch.bool:
        return attention_mask & causal_mask
    causal_bias = torch.zeros((q_len, k_len), dtype=q.dtype, device=q.device)
    causal_bias = causal_bias.masked_fill(~causal_mask, torch.finfo(q.dtype).min)
    return attention_mask.to(device=q.device, dtype=q.dtype) + causal_bias


def _context_parallel_unsupported(name: str, instead: str):
    def unsupported(*args, **kwargs):
        raise NotImplementedError(
            f"difflet.ops.{name} is not implemented on the neuron backend; {instead}"
        )

    unsupported.__name__ = name
    return unsupported


joint_ring_attention = _context_parallel_unsupported(
    "joint_ring_attention", "joint (text + image) sequences are not sharded yet")
joint_ulysses_attention = _context_parallel_unsupported(
    "joint_ulysses_attention", "joint (text + image) sequences are not sharded yet")


def _dense_bhsd(q, k, v, *, scale: float):
    """Unmasked attention on ``[B, H, S, d]`` in the standard layout, through ``attention``'s
    ``attention_cte`` convention (heads folded into the batch, ``tp_q``/``tp_k`` set), as Wan's
    own attention path calls it."""
    b, h, s_q, d = q.shape
    s_k = k.shape[2]
    out = attention(q.reshape(b * h, s_q, d), k.reshape(b * h, s_k, d), v.reshape(b * h, s_k, d),
                    scale=scale, causal=False, tp_q=True, tp_k=True, tp_out=False)
    return out.reshape(b, h, s_q, d)


def ring_attention(q, k, v, *, scale: float, causal: bool = False):
    """Ring context-parallel self-attention over a sequence sharded on the CP group.

    ``q, k, v``: ``[B, H, S/cp, d]`` to ``[B, H, S/cp, d]``. In ``cp`` rounds each rank attends
    its query shard to the K/V block it holds, getting ``(o_r, l_r)`` (output, logsumexp), and
    passes the block to the next rank. Partial results merge in fp32:
    ``l = logaddexp(l, l_r)``, ``o = exp(l_old - l) * o + exp(l_r - l) * o_r``, which is exactly
    the softmax over all ``S`` keys. Non-causal only. Any shard length: the flash kernel's
    logsumexp path needs query lengths in multiples of 128, which ``_attention_with_lse`` pads to.
    """
    if causal:
        raise NotImplementedError("ring_attention supports non-causal attention only")
    from difflet.backends.neuron.ops_impl import collectives
    from difflet.backends.neuron.ops_impl.parallel_mesh import get_axis_rank, get_axis_size

    cp = get_axis_size("cp")
    if cp == 1:
        return _dense_bhsd(q, k, v, scale=scale)
    group, rank = collectives.get_cp_group(), get_axis_rank("cp")
    kv = torch.stack([k, v])  # one hand-off per round for both
    out = lse = None
    for step in range(cp):
        o_r, l_r = _attention_with_lse(q, kv[0], kv[1], scale=scale)
        if out is None:
            out, lse = o_r.float(), l_r
        else:
            merged = torch.logaddexp(lse, l_r)
            out = (out * torch.exp(lse - merged).unsqueeze(-1)
                   + o_r.float() * torch.exp(l_r - merged).unsqueeze(-1))
            lse = merged
        if step < cp - 1:
            kv = collectives.ring_shift(kv, group, cp, rank)
    return out.to(q.dtype)


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
    if q.shape[1] % cp:
        raise ValueError(
            f"cp_mode='ulysses' needs this rank's head count ({q.shape[1]}) divisible by "
            f"cp_degree ({cp}): num_attention_heads must be divisible by tp_degree * cp_degree"
        )
    group = collectives.get_cp_group()
    q, k, v = (collectives.seq_to_heads(t, group, cp) for t in (q, k, v))
    out = _dense_bhsd(q, k, v, scale=scale)
    return collectives.heads_to_seq(out, group, cp)

__all__ = [
    "attention",
    "cross_attention",
    "ring_attention",
    "joint_ring_attention",
    "ulysses_attention",
    "joint_ulysses_attention",
]
