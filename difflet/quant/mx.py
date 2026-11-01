"""MXFP8 (OCP microscaling) quantization, in the exact block membership of Trainium3's MX
tensor-engine path, so the CPU reference and the device compute the same numbers.

Why MX and not per-tensor FP8 on the kernel path: the per-tensor kernel needs the activation
quantized and transposed in the graph before the NKI call, and neuronx-cc 2.28 handles that
chain pathologically when its input is a compiler-internal tensor (1 ms per linear instead of
0.4; an NCC_IINL902 internal error on some shapes). nkilib's ``matmul_mxfp8`` takes the bf16
activation as the kernel input and quantizes inside the kernel: 0.37 ms per 5120x3072x3072
linear against 0.78 ms in bf16, additive with any producer. Block scales also handle the DiT's
activation outliers (a static per-tensor scale saw amax ratios of 20x) without calibration.

The law (nkilib ``quantize_to_mx``, ``nisa.quantize_mx``): 32 values share one E8M0 scale,
``2^(e_max - 7 - 127)`` with ``e_max`` the largest biased fp32 exponent field in the block (so
the block's largest value lands in ``[128, 256)``: the hardware's own, non-OCP rule), and each
value is ``clip(x / scale, -448, 448)`` rounded to ``float8_e4m3fn``.

Which 32 values form a block: the tensor engine contracts over SBUF partitions and the four
bytes packed into each int32, and a block is 8 partitions x 4 bytes of the *swizzled* operand
(``[K/4, F*4]``, nkilib's ``_swizzle``: 512-wide K tiles, each spread over 128 partitions with
the tile's four 128-element quarters interleaved as the four bytes; a K remainder is laid out as
256- and 128-element sub-tiles). So along the logical K axis a block is a fixed, non-contiguous
set of 32 indices. :func:`swizzle_perm` is that order; :func:`quantize_mx` quantizes contiguous
groups of 32 in it and stores the fp8 data back in logical order, so a quantized weight is
``[N, K]`` fp8 plus ``[N, K/32]`` uint8 scales, shardable along N, and along K in units of a
rank's own K range (quantize each rank's shard with its own swizzle, see ``checkpoint.py``).
:func:`kernel_layout` turns a shard into what the kernel reads.
"""

from __future__ import annotations

import torch

from difflet.quant.fp8 import FP8_DTYPE

BLOCK = 32
#: e4m3fn: the largest exponent (biased 134 = 2^7) the block maximum is scaled into.
E4M3_MAX_EXP = 7
E4M3_MAX = 448.0
TILE_K = 512
E8M0_BIAS = 127


def _tile_sizes(k: int) -> list[int]:
    """nkilib's legacy DGT decomposition of K: 512-wide tiles, then a 256 and/or 128 remainder."""
    if k % 128:
        raise ValueError(f"K={k} must be a multiple of 128 for the MX layout")
    tiles = [TILE_K] * (k // TILE_K)
    rem = k % TILE_K
    if rem >= 256:
        tiles.append(256)
        rem -= 256
    if rem >= 128:
        tiles.append(128)
    return tiles


def swizzle_perm(k: int) -> torch.Tensor:
    """``perm[r * 4 + j] = logical k`` of swizzled row ``r`` and byte ``j``: the K order in which
    consecutive groups of 32 are the hardware's MX blocks."""
    perm = []
    k0 = 0
    for tile in _tile_sizes(k):
        sub = tile // 4
        for p in range(sub):
            for j in range(4):
                perm.append(k0 + j * sub + p)
        k0 += tile
    return torch.tensor(perm, dtype=torch.long)


def inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
    return inv


def _block_scales(swizzled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(scale uint8 [..., K/32], scale value fp32 [..., K/32])`` of fp32 ``[..., K]`` in
    swizzled order."""
    exp_field = (swizzled.float().view(torch.int32) >> 23) & 0xFF
    blocks = exp_field.reshape(*swizzled.shape[:-1], -1, BLOCK)
    e_max = blocks.amax(dim=-1)
    scale_u8 = (e_max - E4M3_MAX_EXP).clamp(0, 255)
    value = torch.exp2((scale_u8 - E8M0_BIAS).float())
    return scale_u8.to(torch.uint8), value


def quantize_mx(t: torch.Tensor, perm: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """``[R, K]`` (any float dtype) -> ``(fp8 [R, K] in logical order, uint8 E8M0 scales
    [R, K/32] in block order)``."""
    k = t.shape[-1]
    perm = swizzle_perm(k) if perm is None else perm
    sw = t.float()[..., perm]
    scale_u8, value = _block_scales(sw)
    scaled = sw.reshape(*sw.shape[:-1], -1, BLOCK) / value.unsqueeze(-1)
    q_sw = scaled.clamp(-E4M3_MAX, E4M3_MAX).reshape(sw.shape).to(FP8_DTYPE)
    q = torch.empty_like(q_sw)
    q[..., perm] = q_sw
    return q, scale_u8


def dequantize_mx(q: torch.Tensor, scale_u8: torch.Tensor, perm: torch.Tensor | None = None
                  ) -> torch.Tensor:
    """fp32 ``[R, K]`` in logical order from :func:`quantize_mx`'s outputs."""
    k = q.shape[-1]
    perm = swizzle_perm(k) if perm is None else perm
    value = torch.exp2((scale_u8.float() - E8M0_BIAS))
    sw = q.float()[..., perm].reshape(*q.shape[:-1], -1, BLOCK) * value.unsqueeze(-1)
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    out[..., perm] = sw.reshape(q.shape[:-1] + (k,))
    return out


def fake_quantize_mx(t: torch.Tensor, perm: torch.Tensor | None = None) -> torch.Tensor:
    """``dequantize_mx(quantize_mx(t))``: what the device sees of ``t``."""
    return dequantize_mx(*quantize_mx(t, perm), perm)


def quantize_mx_sharded(t: torch.Tensor, k_shards: int) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`quantize_mx` of ``[R, K]`` in ``k_shards`` equal K ranges, each with its own block
    layout: what a row-parallel weight needs when every rank quantizes (or the kernel loads)
    its own K range. Outputs are in logical order, so a rank's slice of ``[R, K]`` and of
    ``[R, K/32]`` is exactly what :func:`quantize_mx` of that slice gives."""
    k = t.shape[-1]
    if k % k_shards:
        raise ValueError(f"K={k} does not split into {k_shards} shards")
    if k_shards == 1:
        return quantize_mx(t)
    parts = [quantize_mx(chunk) for chunk in t.split(k // k_shards, dim=-1)]
    return torch.cat([p[0] for p in parts], dim=-1), torch.cat([p[1] for p in parts], dim=-1)


def dequantize_mx_sharded(q: torch.Tensor, scale_u8: torch.Tensor, k_shards: int) -> torch.Tensor:
    k = q.shape[-1]
    if k_shards == 1:
        return dequantize_mx(q, scale_u8)
    qs = q.split(k // k_shards, dim=-1)
    ss = scale_u8.split(k // k_shards // BLOCK, dim=-1)
    return torch.cat([dequantize_mx(a, b) for a, b in zip(qs, ss)], dim=-1)


def fake_quantize_mx_sharded(t: torch.Tensor, k_shards: int) -> torch.Tensor:
    return dequantize_mx_sharded(*quantize_mx_sharded(t, k_shards), k_shards)


def kernel_layout(q: torch.Tensor, scale_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A quantized weight shard ``(fp8 [N, K], uint8 [N, K/32])`` -> what nkilib's
    ``matmul_mxfp8`` reads as a pre-quantized rhs: ``data`` int32 ``[K/4, N]`` (the swizzled
    ``[K/4, N*4]`` fp8 bytes packed by fours) and ``scales`` uint8 ``[K/4, N]`` in the hardware
    quadrant layout (four scale rows per 32 partitions, the rest zero)."""
    n, k = q.shape
    perm = swizzle_perm(k).to(q.device)
    q_sw = q[:, perm].reshape(n, k // 4, 4)                       # [N, r, j]
    data = q_sw.permute(1, 0, 2).contiguous().view(torch.int32).squeeze(-1)  # [K/4, N]
    rows = k // 4
    scales = torch.zeros(rows, n, dtype=torch.uint8, device=q.device)
    groups = scale_u8.t()                                           # [K/32, N], g = 8-row group
    g = torch.arange(k // BLOCK, device=q.device)
    scales[(g // 4) * 32 + (g % 4)] = groups
    return data, scales


__all__ = [
    "BLOCK",
    "E4M3_MAX",
    "dequantize_mx",
    "dequantize_mx_sharded",
    "fake_quantize_mx",
    "fake_quantize_mx_sharded",
    "inverse_perm",
    "kernel_layout",
    "quantize_mx",
    "quantize_mx_sharded",
    "swizzle_perm",
]
