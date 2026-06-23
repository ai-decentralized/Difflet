"""Trainium sparse linear operation implementations.

Wraps sparse matmul NKI kernels as TorchXlaKernel for XLA compilation,
following the same pattern as difflet/backends/trainium/ops_impl/mx.py.
"""

from __future__ import annotations

import torch

from difflet.backends.trainium.nki_kernels.sparse_matmul import (
    sparse_matmul_bf16_kernel,
    sparse_matmul_fp8_kernel,
)

_SPARSE_MATMUL_BF16_TORCHXLA_KERNEL = None
_SPARSE_MATMUL_FP8_TORCHXLA_KERNEL = None

_SUPPORTED_MX_DTYPES = ("float8_e4m3fn_x4", "float8_e5m2_x4")


def _get_sparse_matmul_bf16_torchxla_kernel():
    """Singleton TorchXlaKernel for BF16 sparse matmul."""
    global _SPARSE_MATMUL_BF16_TORCHXLA_KERNEL
    if _SPARSE_MATMUL_BF16_TORCHXLA_KERNEL is None:
        from nki.framework.torch_xla import TorchXlaKernel

        _SPARSE_MATMUL_BF16_TORCHXLA_KERNEL = sparse_matmul_bf16_kernel[1]._to_subclass(
            TorchXlaKernel
        )
    return _SPARSE_MATMUL_BF16_TORCHXLA_KERNEL


def _get_sparse_matmul_fp8_torchxla_kernel():
    """Singleton TorchXlaKernel for FP8 sparse matmul."""
    global _SPARSE_MATMUL_FP8_TORCHXLA_KERNEL
    if _SPARSE_MATMUL_FP8_TORCHXLA_KERNEL is None:
        from nki.framework.torch_xla import TorchXlaKernel

        _SPARSE_MATMUL_FP8_TORCHXLA_KERNEL = sparse_matmul_fp8_kernel[1]._to_subclass(
            TorchXlaKernel
        )
    return _SPARSE_MATMUL_FP8_TORCHXLA_KERNEL


def sparse_matmul_bf16(
    compressed_weight: torch.Tensor,
    tags: torch.Tensor,
    activation: torch.Tensor,
    *,
    compress_ratio: int = 4,
) -> torch.Tensor:
    """BF16 16:4 sparse matmul with automatic backend dispatch.

    Args:
        compressed_weight: compressed stationary weight [P, F_stat] BF16/int64
        tags:              packed 4-bit index metadata [P, F_tag] uint16
        activation:        dense moving activation [P, F_mov] BF16
        compress_ratio:    compression ratio (4 for 16:4)

    Returns:
        [F_stat, F_mov] BF16 result
    """
    if compressed_weight.device.type == "cpu":
        raise NotImplementedError(
            "CPU sparse matmul is not yet implemented. "
            "Use Trainium hardware for sparse matmul."
        )

    kernel = _get_sparse_matmul_bf16_torchxla_kernel()
    return kernel(compressed_weight, tags, activation, compress_ratio)


def sparse_matmul_fp8(
    compressed_weight: torch.Tensor,
    tags: torch.Tensor,
    activation: torch.Tensor,
    *,
    compress_ratio: int = 4,
    mx_dtype: str = "float8_e4m3fn_x4",
) -> torch.Tensor:
    """FP8 x4 16:4 sparse matmul with automatic backend dispatch.

    Args:
        compressed_weight: compressed stationary weight [P, F_stat] packed x4 (int32 view)
        tags:              packed 4-bit index metadata [P, F_tag] uint16
        activation:        dense moving activation [P, F_mov] packed x4 (int32 view)
        compress_ratio:    compression ratio (4 for 16:4)
        mx_dtype:          MXFP8 format string

    Returns:
        [F_stat, F_mov] BF16 result
    """
    if compressed_weight.device.type == "cpu":
        raise NotImplementedError(
            "CPU FP8 sparse matmul is not yet implemented. "
            "Use Trainium hardware for sparse matmul."
        )

    if mx_dtype not in _SUPPORTED_MX_DTYPES:
        raise ValueError(
            f"sparse_matmul_fp8 supports {_SUPPORTED_MX_DTYPES}, got {mx_dtype!r}"
        )

    kernel = _get_sparse_matmul_fp8_torchxla_kernel()
    return kernel(compressed_weight, tags, activation, compress_ratio)
