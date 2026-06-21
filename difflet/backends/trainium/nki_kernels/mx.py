"""Single-tile MX kernels for Trainium."""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.tensor_view import TensorView

_P_MAX = 128
_STATIONARY_F_MAX = 128
_MOVING_F_MAX = 512
_SCALE_P_COMPACT = 16
_SCALE_P_PER_QUADRANT = 4
_SBUF_QUADRANT_SIZE = 32
_NUM_SBUF_QUADRANTS = 4

# String -> packed x4 float8 dtype. Both E4M3FN and E5M2 pack four 8-bit
# values into a 32-bit container, so only the SBUF tile dtype changes;
# ``nisa.quantize_mx`` / ``nisa.nc_matmul_mx`` read the MXFP8 format off
# the tile dtype (OCP MX v1.0). Group size / scale law are identical.
_MX_X4_DTYPES = {
    "float8_e4m3fn_x4": nl.float8_e4m3fn_x4,
    "float8_e5m2_x4": nl.float8_e5m2_x4,
}
_DEFAULT_MX_DTYPE = "float8_e4m3fn_x4"
# Precomputed at module import: the NKI trace context cannot resolve the
# ``builtins.sorted`` used inside an f-string at kernel-trace time.
_MX_X4_DTYPE_NAMES = ("float8_e4m3fn_x4", "float8_e5m2_x4")
_INPUT_DTYPES = {
    "bfloat16": nl.bfloat16,
    "float16": nl.float16,
}
_INPUT_DTYPE_NAMES = ("bfloat16", "float16")


def _resolve_mx_x4_dtype(mx_dtype: str):
    """Map an MX dtype string to its NKI packed x4 dtype (trace-time)."""

    kernel_assert(
        mx_dtype in _MX_X4_DTYPES,
        f"mx_dtype must be one of {_MX_X4_DTYPE_NAMES}",
    )
    return _MX_X4_DTYPES[mx_dtype]


def _resolve_input_dtype(input_dtype: str):
    """Map an input dtype string to an NKI dtype (trace-time)."""

    kernel_assert(
        input_dtype in _INPUT_DTYPES,
        f"input_dtype must be one of {_INPUT_DTYPE_NAMES}",
    )
    return _INPUT_DTYPES[input_dtype]


def _load_compact_scale_into_sbuf(scale_hbm, scale_sbuf, free_dim: int) -> None:
    """Scatter compact ``[16, F]`` MX scales into nc_matmul_mx SBUF layout."""

    nisa.memset(dst=scale_sbuf[...], value=0)

    for quadrant_idx in nl.affine_range(_NUM_SBUF_QUADRANTS):
        nisa.dma_copy(
            src=scale_hbm[
                nl.ds(quadrant_idx * _SCALE_P_PER_QUADRANT, _SCALE_P_PER_QUADRANT),
                :free_dim,
            ],
            dst=scale_sbuf[
                nl.ds(quadrant_idx * _SBUF_QUADRANT_SIZE, _SCALE_P_PER_QUADRANT),
                :free_dim,
            ],
        )


def _load_compact_scale_to_sbuf(scale_hbm, free_dim: int):
    """Load compact ``[16, F]`` MX scale rows into nc_matmul_mx SBUF layout."""

    scale_sbuf = nl.ndarray((_P_MAX, free_dim), dtype=nl.uint8, buffer=nl.sbuf)
    _load_compact_scale_into_sbuf(scale_hbm, scale_sbuf, free_dim)
    return scale_sbuf


def _load_compact_scale_tile_into_sbuf(
    scale_hbm: nl.ndarray,
    tile_idx: int,
    scale_sbuf: nl.ndarray,
    free_dim: int,
) -> None:
    """Scatter compact ``[K_tiles, 16, F]`` scale tile into SBUF layout."""

    nisa.memset(dst=scale_sbuf[...], value=0)

    for quadrant_idx in nl.affine_range(_NUM_SBUF_QUADRANTS):
        row_offset = quadrant_idx * _SCALE_P_PER_QUADRANT
        flat_offset = (
            tile_idx * _SCALE_P_COMPACT * free_dim + row_offset * free_dim
        )
        nisa.dma_copy(
            src=scale_hbm.ap(
                pattern=[[free_dim, _SCALE_P_PER_QUADRANT], [1, free_dim]],
                offset=flat_offset,
            ),
            dst=scale_sbuf[
                nl.ds(quadrant_idx * _SBUF_QUADRANT_SIZE, _SCALE_P_PER_QUADRANT),
                :free_dim,
            ],
        )


def _load_mx_data_tile_to_sbuf(
    data_hbm: nl.ndarray,
    tile_idx: int,
    data_sbuf: nl.ndarray,
    free_dim: int,
    mx_x4_dtype,
):
    flat_offset = tile_idx * _P_MAX * free_dim
    nisa.dma_copy(
        dst=data_sbuf,
        src=data_hbm.ap(
            pattern=[[free_dim, _P_MAX], [1, free_dim]],
            offset=flat_offset,
            dtype=mx_x4_dtype,
        ),
    )


def _load_mx_data_native_tile_to_sbuf(
    data_hbm: nl.ndarray,
    tile_idx: int,
    data_sbuf: nl.ndarray,
    mx_x4_dtype,
):
    """Load a ``[K_tiles, 128, F]`` MX tile.

    The AP view is retained for uint32 -> MX x4 dtype reinterpretation; the
    Tier 2 native path removes the moving scale scatter, not this data view.
    """

    _load_mx_data_tile_to_sbuf(
        data_hbm,
        tile_idx,
        data_sbuf,
        _MOVING_F_MAX,
        mx_x4_dtype,
    )


def _load_native_scale_tile_to_sbuf(
    scale_hbm: nl.ndarray,
    tile_idx: int,
    scale_sbuf: nl.ndarray,
) -> None:
    """Load pre-scattered ``[K_tiles, 128, F]`` MX scales with direct DMA."""

    nisa.dma_copy(dst=scale_sbuf, src=scale_hbm[tile_idx, :, :])


def _load_native_data_tile_to_sbuf(
    input_hbm: nl.ndarray,
    tile_idx: int,
    native_sbuf: nl.ndarray,
    input_dtype,
) -> None:
    """Load one activation tile and convert it to native MX stationary layout."""

    input_tile_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX * 4),
        dtype=input_dtype,
        buffer=nl.sbuf,
    )
    tile_offset = tile_idx * _STATIONARY_F_MAX * 4
    nisa.dma_copy(
        dst=input_tile_sbuf,
        src=input_hbm[:, tile_offset : tile_offset + _STATIONARY_F_MAX * 4],
    )

    input_tile_view = TensorView(input_tile_sbuf).reshape_dim(
        dim=1,
        shape=[_STATIONARY_F_MAX, 4],
    )
    transposed_psum = nl.ndarray(
        (_STATIONARY_F_MAX, _P_MAX * 4),
        dtype=input_dtype,
        buffer=nl.psum,
    )

    for q_idx in nl.static_range(4):
        src_view = input_tile_view.slice(dim=2, start=q_idx, end=q_idx + 1).squeeze_dim(
            dim=2
        )
        nisa.nc_transpose(
            dst=transposed_psum[:, q_idx * _P_MAX : (q_idx + 1) * _P_MAX],
            data=src_view.get_view(),
        )

    src_native_view = TensorView(transposed_psum).reshape_dim(
        dim=1,
        shape=[4, _P_MAX],
    ).permute(dims=[0, 2, 1])
    dst_native_view = TensorView(native_sbuf).reshape_dim(
        dim=1,
        shape=[_P_MAX, 4],
    )
    nisa.tensor_copy(dst=dst_native_view.get_view(), src=src_native_view.get_view())


def _store_sbuf_scale_to_compact(scale_sbuf, scale_hbm, free_dim: int) -> None:
    """Store nc_matmul_mx SBUF scale rows back to compact ``[16, F]`` layout."""

    for quadrant_idx in nl.affine_range(_NUM_SBUF_QUADRANTS):
        nisa.dma_copy(
            src=scale_sbuf[
                nl.ds(quadrant_idx * _SBUF_QUADRANT_SIZE, _SCALE_P_PER_QUADRANT),
                :free_dim,
            ],
            dst=scale_hbm[
                nl.ds(quadrant_idx * _SCALE_P_PER_QUADRANT, _SCALE_P_PER_QUADRANT),
                :free_dim,
            ],
        )


@nki.jit
def quantize_mx_e4m3_single_tile_kernel(
    x: nl.ndarray,
    free_dim: int,
    input_dtype: str = "bfloat16",
    mx_dtype: str = _DEFAULT_MX_DTYPE,
):
    """Quantize one ``bfloat16/float16`` tile to MXFP8 (E4M3FN or E5M2).

    ``mx_dtype`` is a trace-time constant selecting the packed x4 output
    format; the kept name is historical (E4M3 default) — the kernel is
    dtype-parametrized per M5.4.1.a.
    """

    mx_x4_dtype = _resolve_mx_x4_dtype(mx_dtype)
    x_dtype = _resolve_input_dtype(input_dtype)
    packed_free_dim = free_dim // 4

    x_sbuf = nl.ndarray((_P_MAX, free_dim), dtype=x_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sbuf, src=x)

    data_sbuf = nl.ndarray(
        (_P_MAX, packed_free_dim),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    scale_sbuf = nl.ndarray((_P_MAX, packed_free_dim), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=data_sbuf, src=x_sbuf, dst_scale=scale_sbuf)

    data_hbm = nl.ndarray((_P_MAX, packed_free_dim), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=data_hbm, src=data_sbuf.view(nl.uint32))

    scale_hbm = nl.ndarray(
        (_SCALE_P_COMPACT, packed_free_dim),
        dtype=nl.uint8,
        buffer=nl.shared_hbm,
    )
    _store_sbuf_scale_to_compact(scale_sbuf, scale_hbm, packed_free_dim)
    return data_hbm, scale_hbm


@nki.jit
def matmul_mx_k_tiles_kernel(
    stationary_mx_data,
    stationary_mx_scale,
    moving_mx_data,
    moving_mx_scale,
    k_tiles: int,
    mx_dtype: str = _DEFAULT_MX_DTYPE,
):
    """Compute one MX output tile while reducing across leading K tiles."""

    mx_x4_dtype = _resolve_mx_x4_dtype(mx_dtype)
    kernel_assert(k_tiles >= 1, "matmul_mx_k_tiles_kernel requires K_tiles >= 1")

    stationary_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    stationary_scale_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    moving_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    moving_scale_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    result_psum = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    _load_mx_data_tile_to_sbuf(
        stationary_mx_data,
        0,
        stationary_sbuf,
        _STATIONARY_F_MAX,
        mx_x4_dtype,
    )
    _load_compact_scale_tile_into_sbuf(
        stationary_mx_scale,
        0,
        stationary_scale_sbuf,
        _STATIONARY_F_MAX,
    )
    _load_mx_data_tile_to_sbuf(
        moving_mx_data,
        0,
        moving_sbuf,
        _MOVING_F_MAX,
        mx_x4_dtype,
    )
    _load_compact_scale_tile_into_sbuf(
        moving_mx_scale,
        0,
        moving_scale_sbuf,
        _MOVING_F_MAX,
    )
    nisa.nc_matmul_mx(
        dst=result_psum,
        stationary=stationary_sbuf,
        moving=moving_sbuf,
        stationary_scale=stationary_scale_sbuf,
        moving_scale=moving_scale_sbuf,
        accumulate=False,
    )

    for k_idx in nl.sequential_range(1, k_tiles):
        _load_mx_data_tile_to_sbuf(
            stationary_mx_data,
            k_idx,
            stationary_sbuf,
            _STATIONARY_F_MAX,
            mx_x4_dtype,
        )
        _load_compact_scale_tile_into_sbuf(
            stationary_mx_scale,
            k_idx,
            stationary_scale_sbuf,
            _STATIONARY_F_MAX,
        )
        _load_mx_data_tile_to_sbuf(
            moving_mx_data,
            k_idx,
            moving_sbuf,
            _MOVING_F_MAX,
            mx_x4_dtype,
        )
        _load_compact_scale_tile_into_sbuf(
            moving_mx_scale,
            k_idx,
            moving_scale_sbuf,
            _MOVING_F_MAX,
        )
        nisa.nc_matmul_mx(
            dst=result_psum,
            stationary=stationary_sbuf,
            moving=moving_sbuf,
            stationary_scale=stationary_scale_sbuf,
            moving_scale=moving_scale_sbuf,
            accumulate=True,
        )

    result_sbuf = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=result_sbuf, src=result_psum)
    result_hbm = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    nisa.dma_copy(dst=result_hbm, src=result_sbuf)
    return result_hbm


@nki.jit
def quantize_mx_linear_activation_kernel(
    input_bf16: nl.ndarray,
    k_tiles: int,
    k_dim: int,
    input_dtype: str = "bfloat16",
    mx_dtype: str = _DEFAULT_MX_DTYPE,
):
    """Quantize a 128xK linear activation once into NCv4 MX stationary tiles."""

    mx_x4_dtype = _resolve_mx_x4_dtype(mx_dtype)
    x_dtype = _resolve_input_dtype(input_dtype)
    kernel_assert(k_tiles >= 1, "quantize_mx_linear_activation_kernel requires K_tiles >= 1")

    stationary_native = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX * 4),
        dtype=x_dtype,
        buffer=nl.sbuf,
    )
    stationary_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    stationary_scale_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )

    data_hbm = nl.ndarray(
        (k_tiles, _P_MAX, _STATIONARY_F_MAX),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
    )
    scale_hbm = nl.ndarray(
        (k_tiles, _SCALE_P_COMPACT, _STATIONARY_F_MAX),
        dtype=nl.uint8,
        buffer=nl.shared_hbm,
    )

    for k_idx in nl.sequential_range(k_tiles):
        _load_native_data_tile_to_sbuf(
            input_bf16,
            k_idx,
            stationary_native,
            x_dtype,
        )
        nisa.quantize_mx(
            dst=stationary_sbuf,
            src=stationary_native,
            dst_scale=stationary_scale_sbuf,
        )
        nisa.dma_copy(dst=data_hbm[k_idx, :, :], src=stationary_sbuf.view(nl.uint32))
        _store_sbuf_scale_to_compact(
            stationary_scale_sbuf,
            scale_hbm[k_idx, :, :],
            _STATIONARY_F_MAX,
        )

    return data_hbm, scale_hbm


@nki.jit
def linear_mx_prequant_kernel(
    input_bf16: nl.ndarray,
    moving_mx_data,
    moving_mx_scale,
    k_tiles: int,
    k_dim: int,
    input_dtype: str = "bfloat16",
    mx_dtype: str = _DEFAULT_MX_DTYPE,
):
    """Compute ``input_bf16 @ prequant_weight_mx`` for one 512-wide N tile."""

    mx_x4_dtype = _resolve_mx_x4_dtype(mx_dtype)
    x_dtype = _resolve_input_dtype(input_dtype)
    kernel_assert(k_tiles >= 1, "linear_mx_prequant_kernel requires K_tiles >= 1")

    stationary_native = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX * 4),
        dtype=x_dtype,
        buffer=nl.sbuf,
    )
    stationary_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    stationary_scale_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    moving_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    moving_scale_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    result_psum = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    _load_native_data_tile_to_sbuf(
        input_bf16,
        0,
        stationary_native,
        x_dtype,
    )
    nisa.quantize_mx(
        dst=stationary_sbuf,
        src=stationary_native,
        dst_scale=stationary_scale_sbuf,
    )
    _load_mx_data_tile_to_sbuf(
        moving_mx_data,
        0,
        moving_sbuf,
        _MOVING_F_MAX,
        mx_x4_dtype,
    )
    _load_compact_scale_tile_into_sbuf(
        moving_mx_scale,
        0,
        moving_scale_sbuf,
        _MOVING_F_MAX,
    )
    nisa.nc_matmul_mx(
        dst=result_psum,
        stationary=stationary_sbuf,
        moving=moving_sbuf,
        stationary_scale=stationary_scale_sbuf,
        moving_scale=moving_scale_sbuf,
        accumulate=False,
    )

    for k_idx in nl.sequential_range(1, k_tiles):
        _load_native_data_tile_to_sbuf(
            input_bf16,
            k_idx,
            stationary_native,
            x_dtype,
        )
        nisa.quantize_mx(
            dst=stationary_sbuf,
            src=stationary_native,
            dst_scale=stationary_scale_sbuf,
        )
        _load_mx_data_tile_to_sbuf(
            moving_mx_data,
            k_idx,
            moving_sbuf,
            _MOVING_F_MAX,
            mx_x4_dtype,
        )
        _load_compact_scale_tile_into_sbuf(
            moving_mx_scale,
            k_idx,
            moving_scale_sbuf,
            _MOVING_F_MAX,
        )
        nisa.nc_matmul_mx(
            dst=result_psum,
            stationary=stationary_sbuf,
            moving=moving_sbuf,
            stationary_scale=stationary_scale_sbuf,
            moving_scale=moving_scale_sbuf,
            accumulate=True,
        )

    result_sbuf = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=result_sbuf, src=result_psum)
    result_hbm = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    nisa.dma_copy(dst=result_hbm, src=result_sbuf)
    return result_hbm


@nki.jit
def linear_mx_prequant_group2_kernel(
    input_bf16: nl.ndarray,
    moving0_mx_data,
    moving0_mx_scale,
    moving1_mx_data,
    moving1_mx_scale,
    k_tiles: int,
    k_dim: int,
    input_dtype: str = "bfloat16",
    mx_dtype: str = _DEFAULT_MX_DTYPE,
):
    """Compute two ``input_bf16 @ prequant_weight_mx`` outputs for one N tile.

    The two linears share the activation load, on-chip native-layout
    transform, and MX quantization. This is the smallest group-fusion probe
    for LTX-2 K/V projections without fusing the whole attention block.
    """

    mx_x4_dtype = _resolve_mx_x4_dtype(mx_dtype)
    x_dtype = _resolve_input_dtype(input_dtype)
    kernel_assert(k_tiles >= 1, "linear_mx_prequant_group2_kernel requires K_tiles >= 1")

    stationary_native = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX * 4),
        dtype=x_dtype,
        buffer=nl.sbuf,
    )
    stationary_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    stationary_scale_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    moving_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    moving_scale_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    result0_psum = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    result1_psum = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    _load_native_data_tile_to_sbuf(
        input_bf16,
        0,
        stationary_native,
        x_dtype,
    )
    nisa.quantize_mx(
        dst=stationary_sbuf,
        src=stationary_native,
        dst_scale=stationary_scale_sbuf,
    )
    _load_mx_data_tile_to_sbuf(
        moving0_mx_data,
        0,
        moving_sbuf,
        _MOVING_F_MAX,
        mx_x4_dtype,
    )
    _load_compact_scale_tile_into_sbuf(
        moving0_mx_scale,
        0,
        moving_scale_sbuf,
        _MOVING_F_MAX,
    )
    nisa.nc_matmul_mx(
        dst=result0_psum,
        stationary=stationary_sbuf,
        moving=moving_sbuf,
        stationary_scale=stationary_scale_sbuf,
        moving_scale=moving_scale_sbuf,
        accumulate=False,
    )
    _load_mx_data_tile_to_sbuf(
        moving1_mx_data,
        0,
        moving_sbuf,
        _MOVING_F_MAX,
        mx_x4_dtype,
    )
    _load_compact_scale_tile_into_sbuf(
        moving1_mx_scale,
        0,
        moving_scale_sbuf,
        _MOVING_F_MAX,
    )
    nisa.nc_matmul_mx(
        dst=result1_psum,
        stationary=stationary_sbuf,
        moving=moving_sbuf,
        stationary_scale=stationary_scale_sbuf,
        moving_scale=moving_scale_sbuf,
        accumulate=False,
    )

    for k_idx in nl.sequential_range(1, k_tiles):
        _load_native_data_tile_to_sbuf(
            input_bf16,
            k_idx,
            stationary_native,
            x_dtype,
        )
        nisa.quantize_mx(
            dst=stationary_sbuf,
            src=stationary_native,
            dst_scale=stationary_scale_sbuf,
        )
        _load_mx_data_tile_to_sbuf(
            moving0_mx_data,
            k_idx,
            moving_sbuf,
            _MOVING_F_MAX,
            mx_x4_dtype,
        )
        _load_compact_scale_tile_into_sbuf(
            moving0_mx_scale,
            k_idx,
            moving_scale_sbuf,
            _MOVING_F_MAX,
        )
        nisa.nc_matmul_mx(
            dst=result0_psum,
            stationary=stationary_sbuf,
            moving=moving_sbuf,
            stationary_scale=stationary_scale_sbuf,
            moving_scale=moving_scale_sbuf,
            accumulate=True,
        )
        _load_mx_data_tile_to_sbuf(
            moving1_mx_data,
            k_idx,
            moving_sbuf,
            _MOVING_F_MAX,
            mx_x4_dtype,
        )
        _load_compact_scale_tile_into_sbuf(
            moving1_mx_scale,
            k_idx,
            moving_scale_sbuf,
            _MOVING_F_MAX,
        )
        nisa.nc_matmul_mx(
            dst=result1_psum,
            stationary=stationary_sbuf,
            moving=moving_sbuf,
            stationary_scale=stationary_scale_sbuf,
            moving_scale=moving_scale_sbuf,
            accumulate=True,
        )

    result0_sbuf = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    result1_sbuf = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=result0_sbuf, src=result0_psum)
    nisa.tensor_copy(dst=result1_sbuf, src=result1_psum)
    result0_hbm = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    result1_hbm = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    nisa.dma_copy(dst=result0_hbm, src=result0_sbuf)
    nisa.dma_copy(dst=result1_hbm, src=result1_sbuf)
    return result0_hbm, result1_hbm


@nki.jit
def linear_mx_prequant_native_weight_kernel(
    input_bf16: nl.ndarray,
    moving_mx_data,
    moving_mx_scale_native,
    k_tiles: int,
    k_dim: int,
    input_dtype: str = "bfloat16",
    mx_dtype: str = _DEFAULT_MX_DTYPE,
):
    """Prequantized linear with moving MX data/scales already in native layout."""

    mx_x4_dtype = _resolve_mx_x4_dtype(mx_dtype)
    x_dtype = _resolve_input_dtype(input_dtype)
    kernel_assert(
        k_tiles >= 1,
        "linear_mx_prequant_native_weight_kernel requires K_tiles >= 1",
    )

    stationary_native = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX * 4),
        dtype=x_dtype,
        buffer=nl.sbuf,
    )
    stationary_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    stationary_scale_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    moving_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    moving_scale_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    result_psum = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    _load_native_data_tile_to_sbuf(
        input_bf16,
        0,
        stationary_native,
        x_dtype,
    )
    nisa.quantize_mx(
        dst=stationary_sbuf,
        src=stationary_native,
        dst_scale=stationary_scale_sbuf,
    )
    _load_mx_data_native_tile_to_sbuf(
        moving_mx_data,
        0,
        moving_sbuf,
        mx_x4_dtype,
    )
    _load_native_scale_tile_to_sbuf(
        moving_mx_scale_native,
        0,
        moving_scale_sbuf,
    )
    nisa.nc_matmul_mx(
        dst=result_psum,
        stationary=stationary_sbuf,
        moving=moving_sbuf,
        stationary_scale=stationary_scale_sbuf,
        moving_scale=moving_scale_sbuf,
        accumulate=False,
    )

    for k_idx in nl.sequential_range(1, k_tiles):
        _load_native_data_tile_to_sbuf(
            input_bf16,
            k_idx,
            stationary_native,
            x_dtype,
        )
        nisa.quantize_mx(
            dst=stationary_sbuf,
            src=stationary_native,
            dst_scale=stationary_scale_sbuf,
        )
        _load_mx_data_native_tile_to_sbuf(
            moving_mx_data,
            k_idx,
            moving_sbuf,
            mx_x4_dtype,
        )
        _load_native_scale_tile_to_sbuf(
            moving_mx_scale_native,
            k_idx,
            moving_scale_sbuf,
        )
        nisa.nc_matmul_mx(
            dst=result_psum,
            stationary=stationary_sbuf,
            moving=moving_sbuf,
            stationary_scale=stationary_scale_sbuf,
            moving_scale=moving_scale_sbuf,
            accumulate=True,
        )

    result_sbuf = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=result_sbuf, src=result_psum)
    result_hbm = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    nisa.dma_copy(dst=result_hbm, src=result_sbuf)
    return result_hbm


@nki.jit
def linear_mx_prequant_group2_native_weight_kernel(
    input_bf16: nl.ndarray,
    moving0_mx_data,
    moving0_mx_scale_native,
    moving1_mx_data,
    moving1_mx_scale_native,
    k_tiles: int,
    k_dim: int,
    input_dtype: str = "bfloat16",
    mx_dtype: str = _DEFAULT_MX_DTYPE,
):
    """Group2 MX linear with both moving operands prepacked in native layout."""

    mx_x4_dtype = _resolve_mx_x4_dtype(mx_dtype)
    x_dtype = _resolve_input_dtype(input_dtype)
    kernel_assert(
        k_tiles >= 1,
        "linear_mx_prequant_group2_native_weight_kernel requires K_tiles >= 1",
    )

    stationary_native = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX * 4),
        dtype=x_dtype,
        buffer=nl.sbuf,
    )
    stationary_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    stationary_scale_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    moving_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    moving_scale_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    result0_psum = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    result1_psum = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    _load_native_data_tile_to_sbuf(
        input_bf16,
        0,
        stationary_native,
        x_dtype,
    )
    nisa.quantize_mx(
        dst=stationary_sbuf,
        src=stationary_native,
        dst_scale=stationary_scale_sbuf,
    )
    _load_mx_data_native_tile_to_sbuf(
        moving0_mx_data,
        0,
        moving_sbuf,
        mx_x4_dtype,
    )
    _load_native_scale_tile_to_sbuf(
        moving0_mx_scale_native,
        0,
        moving_scale_sbuf,
    )
    nisa.nc_matmul_mx(
        dst=result0_psum,
        stationary=stationary_sbuf,
        moving=moving_sbuf,
        stationary_scale=stationary_scale_sbuf,
        moving_scale=moving_scale_sbuf,
        accumulate=False,
    )
    _load_mx_data_native_tile_to_sbuf(
        moving1_mx_data,
        0,
        moving_sbuf,
        mx_x4_dtype,
    )
    _load_native_scale_tile_to_sbuf(
        moving1_mx_scale_native,
        0,
        moving_scale_sbuf,
    )
    nisa.nc_matmul_mx(
        dst=result1_psum,
        stationary=stationary_sbuf,
        moving=moving_sbuf,
        stationary_scale=stationary_scale_sbuf,
        moving_scale=moving_scale_sbuf,
        accumulate=False,
    )

    for k_idx in nl.sequential_range(1, k_tiles):
        _load_native_data_tile_to_sbuf(
            input_bf16,
            k_idx,
            stationary_native,
            x_dtype,
        )
        nisa.quantize_mx(
            dst=stationary_sbuf,
            src=stationary_native,
            dst_scale=stationary_scale_sbuf,
        )
        _load_mx_data_native_tile_to_sbuf(
            moving0_mx_data,
            k_idx,
            moving_sbuf,
            mx_x4_dtype,
        )
        _load_native_scale_tile_to_sbuf(
            moving0_mx_scale_native,
            k_idx,
            moving_scale_sbuf,
        )
        nisa.nc_matmul_mx(
            dst=result0_psum,
            stationary=stationary_sbuf,
            moving=moving_sbuf,
            stationary_scale=stationary_scale_sbuf,
            moving_scale=moving_scale_sbuf,
            accumulate=True,
        )
        _load_mx_data_native_tile_to_sbuf(
            moving1_mx_data,
            k_idx,
            moving_sbuf,
            mx_x4_dtype,
        )
        _load_native_scale_tile_to_sbuf(
            moving1_mx_scale_native,
            k_idx,
            moving_scale_sbuf,
        )
        nisa.nc_matmul_mx(
            dst=result1_psum,
            stationary=stationary_sbuf,
            moving=moving_sbuf,
            stationary_scale=stationary_scale_sbuf,
            moving_scale=moving_scale_sbuf,
            accumulate=True,
        )

    result0_sbuf = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    result1_sbuf = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=result0_sbuf, src=result0_psum)
    nisa.tensor_copy(dst=result1_sbuf, src=result1_psum)
    result0_hbm = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    result1_hbm = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    nisa.dma_copy(dst=result0_hbm, src=result0_sbuf)
    nisa.dma_copy(dst=result1_hbm, src=result1_sbuf)
    return result0_hbm, result1_hbm


@nki.jit
def matmul_mx_single_tile_kernel(
    stationary_mx_data,
    stationary_mx_scale,
    moving_mx_data,
    moving_mx_scale,
    mx_dtype: str = _DEFAULT_MX_DTYPE,
):
    """Compute one NCv4 MXFP8 tile: ``stationary.T @ moving``.

    Inputs are offline-quantized and packed as:
    - ``stationary_mx_data``: ``uint32[128, 128]`` viewed as MXFP8 x4
    - ``stationary_mx_scale``: compact ``uint8[16, 128]``
    - ``moving_mx_data``: ``uint32[128, 512]`` viewed as MXFP8 x4
    - ``moving_mx_scale``: compact ``uint8[16, 512]``

    The result is ``bfloat16[128, 512]`` in shared HBM.
    """

    mx_x4_dtype = _resolve_mx_x4_dtype(mx_dtype)

    stationary_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=stationary_sbuf,
        src=stationary_mx_data.view(mx_x4_dtype),
    )
    stationary_scale_sbuf = _load_compact_scale_to_sbuf(
        stationary_mx_scale,
        _STATIONARY_F_MAX,
    )

    moving_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=mx_x4_dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=moving_sbuf,
        src=moving_mx_data.view(mx_x4_dtype),
    )
    moving_scale_sbuf = _load_compact_scale_to_sbuf(
        moving_mx_scale,
        _MOVING_F_MAX,
    )

    result_psum = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    nisa.nc_matmul_mx(
        dst=result_psum,
        stationary=stationary_sbuf,
        moving=moving_sbuf,
        stationary_scale=stationary_scale_sbuf,
        moving_scale=moving_scale_sbuf,
    )

    result_sbuf = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.tensor_copy(dst=result_sbuf, src=result_psum)
    result_hbm = nl.ndarray(
        (_STATIONARY_F_MAX, _MOVING_F_MAX),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    nisa.dma_copy(dst=result_hbm, src=result_sbuf)
    return result_hbm


__all__ = [
    "linear_mx_prequant_group2_kernel",
    "linear_mx_prequant_group2_native_weight_kernel",
    "linear_mx_prequant_kernel",
    "linear_mx_prequant_native_weight_kernel",
    "matmul_mx_k_tiles_kernel",
    "matmul_mx_single_tile_kernel",
    "quantize_mx_linear_activation_kernel",
    "quantize_mx_e4m3_single_tile_kernel",
]
