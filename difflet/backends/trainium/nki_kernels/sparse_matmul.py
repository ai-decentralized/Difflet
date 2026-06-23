"""NKI kernels for 16:4 structured sparse matrix multiplication.

Wraps ``nc_matmul_sparse`` from the NKI2 private API. Two kernels:
- BF16: standard bfloat16 tiles for both stationary (compressed weight) and
  moving (activation)
- FP8:  float8_e4m3fn_x4 packed tiles for stationary and moving, BF16 output
"""

from __future__ import annotations

import nki
import nki.language as nl
from neuronxcc.nki._private.private_api import nc_matmul_sparse

_P_MAX = 128
_STATIONARY_F_MAX = 128
_MOVING_F_MAX = 512


@nki.jit
def sparse_matmul_bf16_kernel(stationary_compressed, tags, moving, compress_ratio: int):
    """BF16 16:4 sparse matmul: compressed_stationary @ moving -> out.

    stationary_compressed: [P, F_stat] BF16 — compressed weight
                           (P = K/compress_ratio groups along contraction)
    tags:                 [P, F_stat] uint16 — packed 4-bit indices per
                           nonzero element
    moving:               [P, F_mov]  BF16 — activation
    compress_ratio:       int — 4 for 16:4 pattern

    Returns: [F_stat, F_mov] BF16 — result matrix
    """
    P_s, F_s = stationary_compressed.shape
    P_m, F_m = moving.shape
    _P_t, _F_t = tags.shape

    out = nl.ndarray((F_s, F_m), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    stat_sbuf = nl.load(stationary_compressed)
    tags_sbuf = nl.load(tags)
    mov_sbuf = nl.load(moving)

    result = nc_matmul_sparse(
        moving=mov_sbuf,
        stationary=stat_sbuf,
        tags=tags_sbuf,
        compress_ratio=compress_ratio,
    )
    nl.store(out, value=result)
    return out


@nki.jit
def sparse_matmul_fp8_kernel(
    stationary_compressed, tags, moving, compress_ratio: int,
    mx_dtype=None,  # kept for TorchXlaKernel compatibility
):
    """FP8 x4 16:4 sparse matmul: compressed_stationary_fp8x4 @ moving_fp8x4 -> out.

    stationary_compressed: [P, F_stat] uint32 — FP8 x4 packed compressed weight
    tags:                  [P, ...]     uint16 — packed 4-bit indices
    moving:                [P, F_mov]   uint32 — FP8 x4 packed activation
    compress_ratio:        int — 4 for 16:4 pattern

    Returns: [F_stat, F_mov] BF16 — result matrix
    """
    from nki.dtype import float8_e4m3fn_x4 as FP8X4

    P_s, F_s = stationary_compressed.shape
    P_m, F_m = moving.shape
    _P_t, _F_t = tags.shape

    out = nl.ndarray((F_s, F_m), dtype=nl.float32, buffer=nl.shared_hbm)

    stat_sbuf = nl.load(stationary_compressed).view(FP8X4)
    tags_sbuf = nl.load(tags)
    mov_sbuf = nl.load(moving).view(FP8X4)

    result = nc_matmul_sparse(
        moving=mov_sbuf,
        stationary=stat_sbuf,
        tags=tags_sbuf,
        compress_ratio=compress_ratio,
    )
    nl.store(out, value=result)
    return out


__all__ = [
    "sparse_matmul_bf16_kernel",
    "sparse_matmul_fp8_kernel",
]
