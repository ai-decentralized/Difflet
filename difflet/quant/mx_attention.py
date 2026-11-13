"""CPU reference of MXFP8 attention: what the fork of nkilib's ``attention_cte`` computes with
MM1 (QK^T) and/or MM2 (PV) on ``nc_matmul_mx``, so the device probe's output differs from it only
by summation order. It forms P exactly as the kernel does (nkilib 2.14
``core/attention/attention_cte.py``, lines cited as ``:N``):

- Scores: MM1 accumulates in fp32 PSUM (``mm_out_dtype`` fp32, :227) and the evict multiplies by
  ``scale`` into the fp32 ``mm1_masked`` (:2329-2331, ``_emit_row_max_evict`` :3962). Keys outside
  ``[bound_min, bound_max)`` become ``_FLOAT32_MIN`` (:161, ``range_select`` :4282-4299), not
  -inf. The bounded path takes ``scale`` 1.0 (:4286), so its callers pre-scale q in fp32.
- Unbounded keys beyond the real length are never scored: MM1, the row max and ``exponential``
  run on ``num_valid_k`` (:3184, :4093, :3366); by the plan's contract the fork zeroes the last
  P tile's tail before ``quantize_mx``, and V's padding is zero.
- Sections: above 10,240 keys the kernel walks 8,192-key sections (:198-199, :1793-1798). The row
  max is the running max over the sections so far (``tensor_reduce`` then ``minimum`` of the
  negated maxima, :3137-3162), ``E = exp(score - running max)`` (``exponential`` :3381-3387), and
  the correction ``exp(prev_max - cur_max)`` (:3163-3169) rescales the running sum (:3565-3584)
  and the output accumulated so far, which went through ``o`` (q's dtype, :689) between
  sections (:3624, :3663-3675).
- E is written to the bf16 ``exp_sb`` (:2442-2447): that bf16 E is what the fork MX-quantizes per
  512-key tile along keys. The row sum is the fused accumulator of ``exponential``: the fp32 sum
  of the unrounded exp (``exp_partial_sum`` fp32, :2194-2198; the simulator reduces the fp32
  result before the dst cast, ``nki/_backends/simulator/misc.py`` ``exponential``), so it never
  sees P's bf16 or MX rounding. The output is the accumulator times its reciprocal (:3587-3594).

MX operands are quantized along their contraction axis: Q and K along the head dim by
``qk_perm(D)``, P and V along keys per 512-key tile by ``pv_perm(512)`` (:mod:`difflet.quant.mx`
for the law). The perms default to ``swizzle_perm`` until the kernel exports its loaders' own
(ruling BR1). Inputs are taken as given: bf16, as the kernel loads them (:2122-2150), gives the
kernel's numbers; fp32 with MX off is exact attention (PV then takes E in v's dtype), which is how
the tests check the softmax bookkeeping against SDPA.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from difflet.quant.mx import TILE_K, fake_quantize_mx, swizzle_perm

_MASKED = torch.finfo(torch.float32).min  # attention_cte's _FLOAT32_MIN
_SECTION_ABOVE, _SECTION_LEN = 10 * 1024, 8 * 1024


def _fake_quantize_keys(t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """``[..., L]`` zero-padded to whole 512-key tiles, each MX-quantized along its keys."""
    t = F.pad(t, (0, -t.shape[-1] % TILE_K))
    return fake_quantize_mx(t.unflatten(-1, (-1, TILE_K)), perm).flatten(-2)


def mx_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float,
                           bound_min: torch.Tensor | None = None,
                           bound_max: torch.Tensor | None = None, qk_mx: bool, pv_mx: bool,
                           qk_perm: Callable[[int], torch.Tensor] | None = None,
                           pv_perm: Callable[[int], torch.Tensor] | None = None,
                           section_len: int | None = None) -> torch.Tensor:
    """``softmax(scale * q k^T) v`` on ``[BH, S, D]``, query ``i`` attending keys
    ``bound_min[i] <= j < bound_max[i]`` (bounds reshapeable to ``[BH, S_q, 1]``), as the kernel
    computes it (module docstring). ``section_len=None`` sections as the kernel does; a given one
    must be a multiple of 512, as the kernel's are. Returns ``[BH, S_q, D_v]`` in q's dtype."""
    bh, s_q, _ = q.shape
    s_k = k.shape[1]
    if section_len is None:
        section_len = _SECTION_LEN if s_k > _SECTION_ABOVE else s_k
    elif section_len % TILE_K:
        raise ValueError(f"section_len={section_len} must be a multiple of {TILE_K}")
    qf, kf, vf = q.float(), k.float(), v.float()
    if qk_mx:
        perm = (qk_perm or swizzle_perm)(q.shape[-1])
        qf, kf = fake_quantize_mx(qf, perm), fake_quantize_mx(kf, perm)
    if pv_mx:
        tile_perm = (pv_perm or swizzle_perm)(TILE_K)
        vf = _fake_quantize_keys(vf.transpose(1, 2), tile_perm).transpose(1, 2)
    row_max = torch.full((bh, s_q, 1), -torch.inf)
    row_sum = torch.zeros(bh, s_q, 1)
    acc = torch.zeros(bh, s_q, v.shape[-1])
    # shortcut: one section's scores for all of q at once (BH x S_q x 8192 fp32, 10 GB on
    # wan_81f); slice heads or queries in the caller if that is too much.
    for start in range(0, s_k, section_len):
        stop = min(start + section_len, s_k)
        section = (qf @ kf[:, start:stop].transpose(1, 2)) * scale
        if bound_min is not None:
            j = torch.arange(start, stop)
            section = section.masked_fill((j < bound_min.reshape(bh, s_q, 1))
                                          | (j >= bound_max.reshape(bh, s_q, 1)), _MASKED)
        cur_max = torch.maximum(row_max, section.amax(-1, keepdim=True))
        e = torch.exp(section - cur_max)
        correction = torch.exp(row_max - cur_max)
        row_sum = row_sum * correction + e.sum(-1, keepdim=True)
        p = _fake_quantize_keys(e.bfloat16(), tile_perm) if pv_mx else e.to(v.dtype).float()
        acc = acc.to(q.dtype).float() * correction + p @ vf[:, start:start + p.shape[-1]]
        row_max = cur_max
    return (acc * row_sum.reciprocal()).to(q.dtype)


__all__ = ["mx_attention_reference"]
