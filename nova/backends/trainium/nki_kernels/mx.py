"""Single-tile MX kernels for Trainium."""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

_P_MAX = 128
_STATIONARY_F_MAX = 128
_MOVING_F_MAX = 512
_SCALE_P_COMPACT = 16
_SCALE_P_PER_QUADRANT = 4
_SBUF_QUADRANT_SIZE = 32
_NUM_SBUF_QUADRANTS = 4


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
    scale_hbm,
    tile_idx: int,
    scale_sbuf,
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


def _load_mx_data_tile_to_sbuf(data_hbm, tile_idx: int, data_sbuf, free_dim: int):
    flat_offset = tile_idx * _P_MAX * free_dim
    nisa.dma_copy(
        dst=data_sbuf,
        src=data_hbm.ap(
            pattern=[[free_dim, _P_MAX], [1, free_dim]],
            offset=flat_offset,
            dtype=nl.float8_e4m3fn_x4,
        ),
    )


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
def quantize_mx_e4m3_single_tile_kernel(x):
    """Quantize one ``bfloat16/float16`` tile to MXFP8-E4M3FN."""

    kernel_assert(
        x.shape[0] == _P_MAX,
        f"x partition dimension must be {_P_MAX}",
    )
    free_dim = x.shape[1]
    packed_free_dim = free_dim // 4

    x_sbuf = nl.ndarray((_P_MAX, free_dim), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sbuf, src=x)

    data_sbuf = nl.ndarray(
        (_P_MAX, packed_free_dim),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
    )
    scale_sbuf = nl.ndarray(data_sbuf.shape, dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=data_sbuf, src=x_sbuf, dst_scale=scale_sbuf)

    data_hbm = nl.ndarray(data_sbuf.shape, dtype=nl.uint32, buffer=nl.shared_hbm)
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
):
    """Compute one MX output tile while reducing across leading K tiles."""

    k_tiles = stationary_mx_data.shape[0]
    kernel_assert(k_tiles >= 1, "matmul_mx_k_tiles_kernel requires K_tiles >= 1")
    kernel_assert(
        stationary_mx_data.shape[1:] == (_P_MAX, _STATIONARY_F_MAX),
        f"stationary_mx_data tile shape must be ({_P_MAX}, {_STATIONARY_F_MAX})",
    )
    kernel_assert(
        stationary_mx_scale.shape == (
            k_tiles,
            _SCALE_P_COMPACT,
            _STATIONARY_F_MAX,
        ),
        "stationary_mx_scale shape must be "
        f"(K_tiles, {_SCALE_P_COMPACT}, {_STATIONARY_F_MAX})",
    )
    kernel_assert(
        moving_mx_data.shape == (k_tiles, _P_MAX, _MOVING_F_MAX),
        f"moving_mx_data shape must be (K_tiles, {_P_MAX}, {_MOVING_F_MAX})",
    )
    kernel_assert(
        moving_mx_scale.shape == (k_tiles, _SCALE_P_COMPACT, _MOVING_F_MAX),
        "moving_mx_scale shape must be "
        f"(K_tiles, {_SCALE_P_COMPACT}, {_MOVING_F_MAX})",
    )

    stationary_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
    )
    stationary_scale_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=nl.uint8,
        buffer=nl.sbuf,
    )
    moving_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=nl.float8_e4m3fn_x4,
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

    result_sbuf = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result_sbuf, src=result_psum)
    result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=result_hbm, src=result_sbuf)
    return result_hbm


@nki.jit
def matmul_mx_single_tile_kernel(
    stationary_mx_data,
    stationary_mx_scale,
    moving_mx_data,
    moving_mx_scale,
):
    """Compute one NCv4 MXFP8 tile: ``stationary.T @ moving``.

    Inputs are offline-quantized and packed as:
    - ``stationary_mx_data``: ``uint32[128, 128]`` viewed as MXFP8 x4
    - ``stationary_mx_scale``: compact ``uint8[16, 128]``
    - ``moving_mx_data``: ``uint32[128, 512]`` viewed as MXFP8 x4
    - ``moving_mx_scale``: compact ``uint8[16, 512]``

    The result is ``bfloat16[128, 512]`` in shared HBM.
    """

    kernel_assert(
        stationary_mx_data.shape == (_P_MAX, _STATIONARY_F_MAX),
        f"stationary_mx_data must be ({_P_MAX}, {_STATIONARY_F_MAX})",
    )
    kernel_assert(
        stationary_mx_scale.shape == (_SCALE_P_COMPACT, _STATIONARY_F_MAX),
        f"stationary_mx_scale must be ({_SCALE_P_COMPACT}, {_STATIONARY_F_MAX})",
    )
    kernel_assert(
        moving_mx_data.shape == (_P_MAX, _MOVING_F_MAX),
        f"moving_mx_data must be ({_P_MAX}, {_MOVING_F_MAX})",
    )
    kernel_assert(
        moving_mx_scale.shape == (_SCALE_P_COMPACT, _MOVING_F_MAX),
        f"moving_mx_scale must be ({_SCALE_P_COMPACT}, {_MOVING_F_MAX})",
    )

    stationary_sbuf = nl.ndarray(
        (_P_MAX, _STATIONARY_F_MAX),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=stationary_sbuf,
        src=stationary_mx_data.view(nl.float8_e4m3fn_x4),
    )
    stationary_scale_sbuf = _load_compact_scale_to_sbuf(
        stationary_mx_scale,
        _STATIONARY_F_MAX,
    )

    moving_sbuf = nl.ndarray(
        (_P_MAX, _MOVING_F_MAX),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=moving_sbuf,
        src=moving_mx_data.view(nl.float8_e4m3fn_x4),
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

    result_sbuf = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result_sbuf, src=result_psum)
    result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=result_hbm, src=result_sbuf)
    return result_hbm


__all__ = [
    "matmul_mx_k_tiles_kernel",
    "matmul_mx_single_tile_kernel",
    "quantize_mx_e4m3_single_tile_kernel",
]
