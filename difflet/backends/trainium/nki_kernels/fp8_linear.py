"""FP8 (W8A8) linear layer as one NKI kernel: ``y = dequant(fp8(x) @ W_fp8) + bias``.

The XLA-lowered FP8 linear runs the activation reduction (dynamic scales), the quantize and the
dequantize as separate passes around the matmul, with spills to HBM. This kernel keeps every
step on chip, the Trainium counterpart of a GPU FP8 GEMM with a fused epilogue:

  for each output column block (fp8 weight block resident in SBUF):
    for each 128-token tile of x:
      load x K-major with a DMA transpose (K onto partitions; the "_te" v1 kernels instead load
        x [128, K] token-major and transpose 128x128 fp8 blocks on the tensor engine)
      scale:  static  -> the calibrated per-tensor input scale
              token   -> per-row absmax over K on the vector engine (one row = one token)
      quantize to fp8 (static: reciprocal scale + clamp; token: per-token reciprocal broadcast)
      fp8 matmul, double-row mode, fp32 accumulation in PSUM
      PSUM -> SBUF eviction applies (input scale x weight scale) and the bias in one instruction
      write bf16 rows

Layouts: ``x`` [S, K] bf16 (token-major, as the model produces it), ``w_t`` [K, N] fp8 e4m3
(the transposed weight), ``w_scale`` / ``in_scale`` [1, 1] float32, ``bias`` [1, N] bf16.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_helpers import get_program_sharding_info
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

_P = 128
_MOVING = 512
_FP8 = nl.float8_e4m3
_FP8_MAX = 240.0
_MIN_SCALE = 1.0 / (240.0 * 512.0)
# Same margin as difflet.quant.fp8.ACT_SCALE_MARGIN: keeps the rounded absmax below 240.
_TOKEN_MARGIN = 1.0 + 2.0 ** -7
_FP8_PSUM_STEP = 2  # nc_transpose of a 1-byte dtype writes PSUM with a step of 2 (HW constraint)


def _broadcast_scalar(src_hbm):
    """[1, 1] float32 HBM scalar -> [128, 1] SBUF vector (one value per partition)."""
    vec = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vec[0:1, 0:1], src=src_hbm[0:1, 0:1])
    stream_shuffle_broadcast(vec, vec)
    return vec


_N_BLOCK = 2048  # output columns per resident fp8 weight block (SBUF budget at K <= 5120)


@nki.jit
def fp8_linear_static_kernel(x, w_t, w_scale, in_scale, bias):
    """Calibrated per-tensor input scale (``in_scale``); x loaded K-major by DMA transpose."""
    return _fp8_linear(x, w_t, w_scale, in_scale, bias, "static", "dma")


@nki.jit
def fp8_linear_token_kernel(x, w_t, w_scale, in_scale, bias):
    """Per-token dynamic input scale (``in_scale`` is ignored); x loaded row-major for the scales
    and K-major by DMA transpose for the matmul."""
    return _fp8_linear(x, w_t, w_scale, in_scale, bias, "token", "dma")


@nki.jit
def fp8_linear_static_te_kernel(x, w_t, w_scale, in_scale, bias):
    """v1: static scale, x transposed on the tensor engine (kept for comparison)."""
    return _fp8_linear(x, w_t, w_scale, in_scale, bias, "static", "te")


@nki.jit
def fp8_linear_token_te_kernel(x, w_t, w_scale, in_scale, bias):
    """v1: per-token scale, x transposed on the tensor engine (kept for comparison)."""
    return _fp8_linear(x, w_t, w_scale, in_scale, bias, "token", "te")


def fp8_linear_kernel(x, w_t, w_scale, in_scale, bias, mode="static"):
    """Mode-dispatching convenience wrapper (CPU simulation / tests)."""
    kernel = fp8_linear_static_kernel if mode == "static" else fp8_linear_token_kernel
    return kernel(x, w_t, w_scale, in_scale, bias)


def _fp8_linear(x, w_t, w_scale, in_scale, bias, mode, xpose):
    n_block = _N_BLOCK
    S, K = x.shape
    _, N = w_t.shape
    KT = K // _P
    has_bias = bias is not None
    out = nl.ndarray((S, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    _, n_prgs, prg_id = get_program_sharding_info()

    ws_vec = _broadcast_scalar(w_scale)
    inv_in = ws_vec
    comb_static = ws_vec
    if mode == "static":
        in_vec = _broadcast_scalar(in_scale)
        inv_in = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_in, data=in_vec)
        comb_static = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=comb_static, data1=in_vec, data2=ws_vec, op=nl.multiply)
    ones = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)

    for nb0 in range(0, N, n_block):
        nbs = min(n_block, N - nb0)
        w_sb = nl.ndarray((_P, KT, nbs), dtype=_FP8, buffer=nl.sbuf)
        for kt in range(KT):
            nisa.dma_copy(dst=w_sb[:, kt, :], src=w_t[kt * _P:(kt + 1) * _P, nb0:nb0 + nbs])
        b_sb = ws_vec
        if has_bias:
            b_sb = nl.ndarray((_P, nbs), dtype=nl.float32, buffer=nl.sbuf)
            b_row = nl.ndarray((1, nbs), dtype=bias.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b_row, src=bias[0:1, nb0:nb0 + nbs])
            nisa.tensor_copy(dst=b_sb[0:1, :], src=b_row)
            stream_shuffle_broadcast(b_sb, b_sb)

        # Launched as kernel[2] under LNC=2 (one program per physical core), the token tiles are
        # split across the programs; without a grid this is (1 program, id 0) = all tiles.
        num_full = S // _P
        tail = S - num_full * _P
        for st in range(prg_id, num_full, n_prgs):
            _token_tile(x, out, w_sb, b_sb, ws_vec, inv_in, comb_static, st * _P, _P, K, KT, nb0, nbs,
                        has_bias, mode, xpose, ones)
        if tail > 0 and prg_id == num_full % n_prgs:
            _token_tile(x, out, w_sb, b_sb, ws_vec, inv_in, comb_static, num_full * _P, tail, K, KT, nb0, nbs,
                        has_bias, mode, xpose, ones)
    return out


def _token_tile(x, out, w_sb, b_sb, ws_vec, inv_in, comb_static, s0, ss, K, KT, nb0, nbs, has_bias, mode,
                xpose, ones):
    """One 128-token tile (ss <= 128 rows starting at s0) against one resident weight block.
    Produces xT [128 (k), KT, ss] fp8 and the per-row dequant scale ``comb``, then matmuls."""
    if xpose == "dma":
        xT, comb = _quantize_tile_dma(x, ws_vec, inv_in, comb_static, s0, ss, K, KT, mode, ones)
    else:
        xT, comb = _quantize_tile_te(x, ws_vec, inv_in, comb_static, s0, ss, K, KT, mode)

    o_sb = nl.ndarray((_P, nbs), dtype=nl.bfloat16, buffer=nl.sbuf)
    for nt0 in range(0, nbs, _MOVING):
        nts = min(_MOVING, nbs - nt0)
        ps = nl.ndarray((_P, nts), dtype=nl.float32, buffer=nl.psum)
        n_pairs = KT // 2
        for kp in range(n_pairs):
            nisa.nc_matmul(
                ps[:ss, :nts],
                xT[:, 2 * kp:2 * kp + 2, :ss],
                w_sb[:, 2 * kp:2 * kp + 2, nt0:nt0 + nts],
                perf_mode=nisa.matmul_perf_mode.double_row,
                accumulate=(kp > 0),
            )
        if KT % 2:
            nisa.nc_matmul(ps[:ss, :nts], xT[:, KT - 1, :ss], w_sb[:, KT - 1, nt0:nt0 + nts],
                           accumulate=(n_pairs > 0))
        if has_bias:
            nisa.scalar_tensor_tensor(dst=o_sb[:ss, nt0:nt0 + nts], data=ps[:ss, :nts], op0=nl.multiply,
                                      operand0=comb[:ss, :], op1=nl.add, operand1=b_sb[:ss, nt0:nt0 + nts])
        else:
            nisa.activation(dst=o_sb[:ss, nt0:nt0 + nts], op=nl.copy, data=ps[:ss, :nts],
                            scale=comb[:ss, :])
    nisa.dma_copy(dst=out[s0:s0 + ss, nb0:nb0 + nbs], src=o_sb[:ss, :nbs])


def _quantize_tile_te(x, ws_vec, inv_in, comb_static, s0, ss, K, KT, mode):
    """v1: row-major load, quantize per row, transpose 128x128 fp8 blocks on the tensor engine."""
    x_sb = nl.ndarray((_P, K), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb[:ss, :], src=x[s0:s0 + ss, :])

    q8 = nl.ndarray((_P, K), dtype=_FP8, buffer=nl.sbuf)
    if mode == "static":
        scaled = nl.ndarray((_P, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=scaled[:ss, :], op=nl.copy, data=x_sb[:ss, :], scale=inv_in[:ss, :])
        nisa.tensor_scalar(dst=q8[:ss, :], data=scaled[:ss, :], op0=nl.minimum, operand0=_FP8_MAX,
                           op1=nl.maximum, operand1=-_FP8_MAX)
        comb = comb_static
    else:
        absmax = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        abs_sb = nl.ndarray((_P, K), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar_reduce(dst=abs_sb[:ss, :], data=x_sb[:ss, :], op0=nl.abs, operand0=0.0,
                                  reduce_op=nl.maximum, reduce_res=absmax[:ss, :])
        dq = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dq[:ss, :], data=absmax[:ss, :], op0=nl.multiply,
                           operand0=_TOKEN_MARGIN / _FP8_MAX, op1=nl.maximum, operand1=_MIN_SCALE)
        inv = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv[:ss, :], data=dq[:ss, :])
        nisa.activation(dst=q8[:ss, :], op=nl.copy, data=x_sb[:ss, :], scale=inv[:ss, :])
        comb = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=comb[:ss, :], data1=dq[:ss, :], data2=ws_vec[:ss, :], op=nl.multiply)

    xT = nl.ndarray((_P, KT, _P), dtype=_FP8, buffer=nl.sbuf)
    for kt in range(KT):
        xp = nl.ndarray((_P, ss, _FP8_PSUM_STEP), dtype=_FP8, buffer=nl.psum)
        nisa.nc_transpose(
            dst=xp.ap([[ss * _FP8_PSUM_STEP, _P], [_FP8_PSUM_STEP, ss]], offset=0),
            data=q8[:ss, kt * _P:(kt + 1) * _P],
        )
        nisa.tensor_copy(dst=xT[:, kt, :ss], src=xp[:, :ss, 0])
    return xT, comb


def _quantize_tile_dma(x, ws_vec, inv_in, comb_static, s0, ss, K, KT, mode, ones):
    """v2: DMA-transpose the bf16 tile straight into K-major SBUF (tensor engine left to the
    matmul), quantize there. Per-token mode also loads the tile row-major for the row scales and
    broadcasts 1/scale across partitions (row-of-ones scaled per row, one small transpose)."""
    xT_bf = nl.ndarray((_P, KT, _P), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_transpose(
        dst=xT_bf.ap(pattern=[[KT * _P, _P], [1, 1], [_P, KT], [1, ss]], offset=0),
        src=x.ap(pattern=[[K, ss], [1, 1], [_P, KT], [1, _P]], offset=s0 * K),
    )
    xT = nl.ndarray((_P, KT, _P), dtype=_FP8, buffer=nl.sbuf)
    if mode == "static":
        scaled = nl.ndarray((_P, KT, _P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=scaled[:, :, :ss], op=nl.copy, data=xT_bf[:, :, :ss], scale=inv_in)
        nisa.tensor_scalar(dst=xT[:, :, :ss], data=scaled[:, :, :ss], op0=nl.minimum, operand0=_FP8_MAX,
                           op1=nl.maximum, operand1=-_FP8_MAX)
        comb = comb_static
    else:
        x_sb = nl.ndarray((_P, K), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb[:ss, :], src=x[s0:s0 + ss, :])
        absmax = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        abs_sb = nl.ndarray((_P, K), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar_reduce(dst=abs_sb[:ss, :], data=x_sb[:ss, :], op0=nl.abs, operand0=0.0,
                                  reduce_op=nl.maximum, reduce_res=absmax[:ss, :])
        dq = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dq[:ss, :], data=absmax[:ss, :], op0=nl.multiply,
                           operand0=_TOKEN_MARGIN / _FP8_MAX, op1=nl.maximum, operand1=_MIN_SCALE)
        inv = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv[:ss, :], data=dq[:ss, :])
        # rowmat[s, j] = inv[s]  ->  transposed: invB[j, s] = inv[s] (same on every partition j)
        rowmat = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rowmat[:ss, :], op=nl.copy, data=ones[:ss, :], scale=inv[:ss, :])
        inv_b = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=inv_b[:, :ss], data=rowmat[:ss, :])
        for kt in range(KT):
            nisa.tensor_tensor(dst=xT[:, kt, :ss], data1=xT_bf[:, kt, :ss], data2=inv_b[:, :ss], op=nl.multiply)
        comb = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=comb[:ss, :], data1=dq[:ss, :], data2=ws_vec[:ss, :], op=nl.multiply)
    return xT, comb
