"""Microscaling (MX) operation surface."""

from __future__ import annotations

import torch

from difflet.ops._dispatch import load_backend_attr


def quantize_mx(
    x: torch.Tensor,
    *,
    dtype: str = "float8_e4m3fn_x4",
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP16 tensor to packed MX data plus E8M0 scales."""

    impl = load_backend_attr("mx", "quantize_mx")
    return impl(x, dtype=dtype, group_size=group_size)


def dequantize_mx(
    data: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: str = "float8_e4m3fn_x4",
    output_dtype: torch.dtype = torch.bfloat16,
    group_size: int = 32,
) -> torch.Tensor:
    """Dequantize packed MX data back to a regular torch tensor."""

    impl = load_backend_attr("mx", "dequantize_mx")
    return impl(
        data,
        scale,
        dtype=dtype,
        output_dtype=output_dtype,
        group_size=group_size,
    )


def matmul_mx(
    a_mx: torch.Tensor,
    a_scale: torch.Tensor,
    b_mx: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    dtype: str = "float8_e4m3fn_x4",
    out_dtype: torch.dtype = torch.bfloat16,
    accumulate: bool = False,
    group_size: int = 32,
) -> torch.Tensor:
    """Compute ``A @ B`` from packed MX inputs."""

    impl = load_backend_attr("mx", "matmul_mx")
    return impl(
        a_mx,
        a_scale,
        b_mx,
        b_scale,
        dtype=dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        group_size=group_size,
    )


def linear_mx(
    input_bf16: torch.Tensor,
    weight_k_n_bf16: torch.Tensor,
    bias_bf16: torch.Tensor | None = None,
    *,
    dtype: str = "float8_e4m3fn_x4",
    out_dtype: torch.dtype = torch.bfloat16,
    group_size: int = 32,
) -> torch.Tensor:
    """Compute a logical Linear from BF16/FP16 tensors using MX K-tile matmul."""

    impl = load_backend_attr("mx", "linear_mx")
    return impl(
        input_bf16,
        weight_k_n_bf16,
        bias_bf16,
        dtype=dtype,
        out_dtype=out_dtype,
        group_size=group_size,
    )


__all__ = ["dequantize_mx", "linear_mx", "matmul_mx", "quantize_mx"]
