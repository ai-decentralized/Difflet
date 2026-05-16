"""Trainium MX operation implementations."""

from __future__ import annotations

import torch

from nova.backends.cpu.ops_impl import mx as cpu_mx
from nova.backends.trainium.nki_kernels.mx import (
    matmul_mx_k_tiles_kernel,
    matmul_mx_single_tile_kernel,
    quantize_mx_e4m3_single_tile_kernel,
)


_SINGLE_TILE_STATIONARY_SHAPE = (128, 128)
_SINGLE_TILE_MOVING_SHAPE = (128, 512)
_SINGLE_TILE_STATIONARY_SCALE_SHAPE = (16, 128)
_SINGLE_TILE_MOVING_SCALE_SHAPE = (16, 512)


def quantize_mx(
    x: torch.Tensor,
    *,
    dtype: str = "float8_e4m3fn_x4",
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.device.type == "cpu":
        return cpu_mx.quantize_mx(x, dtype=dtype, group_size=group_size)
    _validate_quantize_single_tile_input(x, dtype=dtype, group_size=group_size)
    data, scale = quantize_mx_e4m3_single_tile_kernel[1](x)
    return data, scale


def dequantize_mx(
    data: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: str = "float8_e4m3fn_x4",
    output_dtype: torch.dtype = torch.bfloat16,
    group_size: int = 32,
) -> torch.Tensor:
    if data.device.type == "cpu":
        return cpu_mx.dequantize_mx(
            data,
            scale,
            dtype=dtype,
            output_dtype=output_dtype,
            group_size=group_size,
        )
    del data, scale, dtype, output_dtype, group_size
    raise NotImplementedError("Trainium dequantize_mx is CPU-reference only")


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
    if a_mx.device.type == "cpu":
        if a_mx.ndim == 3:
            return cpu_mx.matmul_mx_k_tiles_reference(
                a_mx,
                a_scale,
                b_mx,
                b_scale,
                dtype=dtype,
                out_dtype=out_dtype,
                group_size=group_size,
            )
        return cpu_mx.matmul_mx(
            a_mx,
            a_scale,
            b_mx,
            b_scale,
            dtype=dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
            group_size=group_size,
        )

    del accumulate
    if a_mx.ndim == 3:
        _validate_k_tiles_inputs(
            a_mx,
            a_scale,
            b_mx,
            b_scale,
            dtype=dtype,
            out_dtype=out_dtype,
            group_size=group_size,
        )
        return matmul_mx_k_tiles_kernel[1](a_mx, a_scale, b_mx, b_scale)

    _validate_single_tile_inputs(
        a_mx,
        a_scale,
        b_mx,
        b_scale,
        dtype=dtype,
        out_dtype=out_dtype,
        group_size=group_size,
    )
    return matmul_mx_single_tile_kernel[1](a_mx, a_scale, b_mx, b_scale)


def linear_mx(
    input_bf16: torch.Tensor,
    weight_k_n_bf16: torch.Tensor,
    bias_bf16: torch.Tensor | None = None,
    *,
    dtype: str = "float8_e4m3fn_x4",
    out_dtype: torch.dtype = torch.bfloat16,
    group_size: int = 32,
) -> torch.Tensor:
    if input_bf16.device.type == "cpu":
        return cpu_mx.linear_mx(
            input_bf16,
            weight_k_n_bf16,
            bias_bf16,
            dtype=dtype,
            out_dtype=out_dtype,
            group_size=group_size,
        )

    _validate_linear_mx_inputs(
        input_bf16,
        weight_k_n_bf16,
        bias_bf16,
        dtype=dtype,
        out_dtype=out_dtype,
        group_size=group_size,
    )
    if weight_k_n_bf16.device != input_bf16.device:
        weight_k_n_bf16 = weight_k_n_bf16.to(input_bf16.device)
    if bias_bf16 is not None and bias_bf16.device != input_bf16.device:
        bias_bf16 = bias_bf16.to(input_bf16.device)

    outputs = []
    for n_start in range(0, weight_k_n_bf16.shape[1], 512):
        n_end = n_start + 512
        weight_tile_n = weight_k_n_bf16[:, n_start:n_end].contiguous()
        bias_tile_n = (
            bias_bf16[n_start:n_end].contiguous() if bias_bf16 is not None else None
        )
        outputs.append(
            _linear_mx_single_n_tile(
                input_bf16,
                weight_tile_n,
                bias_tile_n,
                dtype=dtype,
                out_dtype=out_dtype,
                group_size=group_size,
            )
        )
    return torch.cat(outputs, dim=1)


def _linear_mx_single_n_tile(
    input_bf16: torch.Tensor,
    weight_k_n_bf16: torch.Tensor,
    bias_bf16: torch.Tensor | None,
    *,
    dtype: str,
    out_dtype: torch.dtype,
    group_size: int,
) -> torch.Tensor:
    stationary_tiles = []
    stationary_scales = []
    moving_tiles = []
    moving_scales = []
    for tile_idx in range(input_bf16.shape[1] // 512):
        k_start = tile_idx * 512
        k_end = k_start + 512
        input_tile = input_bf16[:, k_start:k_end].contiguous()
        weight_tile = weight_k_n_bf16[k_start:k_end, :].contiguous()

        activation_native = (
            input_tile.reshape(128, 128, 4)
            .permute(1, 0, 2)
            .reshape(128, 128 * 4)
            .contiguous()
        )
        weight_native = (
            weight_tile.reshape(128, 4, 512)
            .permute(0, 2, 1)
            .reshape(128, 512 * 4)
            .contiguous()
        )
        stationary_mx, stationary_scale = quantize_mx(
            activation_native,
            dtype=dtype,
            group_size=group_size,
        )
        moving_mx, moving_scale = quantize_mx(
            weight_native,
            dtype=dtype,
            group_size=group_size,
        )
        stationary_tiles.append(stationary_mx)
        stationary_scales.append(stationary_scale)
        moving_tiles.append(moving_mx)
        moving_scales.append(moving_scale)

    out = matmul_mx(
        torch.stack(stationary_tiles, dim=0),
        torch.stack(stationary_scales, dim=0),
        torch.stack(moving_tiles, dim=0),
        torch.stack(moving_scales, dim=0),
        dtype=dtype,
        out_dtype=out_dtype,
        group_size=group_size,
    )
    if bias_bf16 is not None:
        out = out + bias_bf16
    return out.to(out_dtype)


def _validate_single_tile_inputs(
    a_mx: torch.Tensor,
    a_scale: torch.Tensor,
    b_mx: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    dtype: str,
    out_dtype: torch.dtype,
    group_size: int,
) -> None:
    if dtype != "float8_e4m3fn_x4":
        raise NotImplementedError(f"Trainium matmul_mx supports only MXFP8 E4M3FN, got {dtype!r}")
    if out_dtype is not torch.bfloat16:
        raise NotImplementedError(f"Trainium matmul_mx supports only BF16 output, got {out_dtype}")
    if group_size != 32:
        raise NotImplementedError(
            f"Trainium matmul_mx supports only group_size=32, got {group_size}"
        )
    if a_mx.dtype not in (torch.int32, torch.uint32) or b_mx.dtype not in (
        torch.int32,
        torch.uint32,
    ):
        raise TypeError("Trainium matmul_mx expects 32-bit packed MX data")
    if a_scale.dtype != torch.uint8 or b_scale.dtype != torch.uint8:
        raise TypeError("Trainium matmul_mx expects uint8 MX scales")
    if tuple(a_mx.shape) != _SINGLE_TILE_STATIONARY_SHAPE:
        raise ValueError(
            "single-tile stationary data must be "
            f"{_SINGLE_TILE_STATIONARY_SHAPE}, got {tuple(a_mx.shape)}"
        )
    if tuple(a_scale.shape) != _SINGLE_TILE_STATIONARY_SCALE_SHAPE:
        raise ValueError(
            "single-tile stationary scale must be "
            f"{_SINGLE_TILE_STATIONARY_SCALE_SHAPE}, got {tuple(a_scale.shape)}"
        )
    if tuple(b_mx.shape) != _SINGLE_TILE_MOVING_SHAPE:
        raise ValueError(
            f"single-tile moving data must be {_SINGLE_TILE_MOVING_SHAPE}, got {tuple(b_mx.shape)}"
        )
    if tuple(b_scale.shape) != _SINGLE_TILE_MOVING_SCALE_SHAPE:
        raise ValueError(
            "single-tile moving scale must be "
            f"{_SINGLE_TILE_MOVING_SCALE_SHAPE}, got {tuple(b_scale.shape)}"
        )


def _validate_k_tiles_inputs(
    a_mx: torch.Tensor,
    a_scale: torch.Tensor,
    b_mx: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    dtype: str,
    out_dtype: torch.dtype,
    group_size: int,
) -> None:
    if a_mx.shape[0] < 1:
        raise ValueError("K-tile matmul requires at least one tile")
    for tile_idx in range(a_mx.shape[0]):
        _validate_single_tile_inputs(
            a_mx[tile_idx],
            a_scale[tile_idx],
            b_mx[tile_idx],
            b_scale[tile_idx],
            dtype=dtype,
            out_dtype=out_dtype,
            group_size=group_size,
        )


def _validate_quantize_single_tile_input(
    x: torch.Tensor,
    *,
    dtype: str,
    group_size: int,
) -> None:
    if dtype != "float8_e4m3fn_x4":
        raise NotImplementedError(f"Trainium quantize_mx supports only MXFP8 E4M3FN, got {dtype!r}")
    if group_size != 32:
        raise NotImplementedError(
            f"Trainium quantize_mx supports only group_size=32, got {group_size}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Trainium quantize_mx expects BF16 or FP16 input, got {x.dtype}")
    if x.ndim != 2 or x.shape[0] != 128 or x.shape[1] not in (512, 2048):
        raise ValueError(
            "single-tile quantize_mx input must be (128, 512) or "
            f"(128, 2048), got {tuple(x.shape)}"
        )


def _validate_linear_mx_inputs(
    input_bf16: torch.Tensor,
    weight_k_n_bf16: torch.Tensor,
    bias_bf16: torch.Tensor | None,
    *,
    dtype: str,
    out_dtype: torch.dtype,
    group_size: int,
) -> None:
    if dtype != "float8_e4m3fn_x4":
        raise NotImplementedError(f"Trainium linear_mx supports only MXFP8 E4M3FN, got {dtype!r}")
    if out_dtype is not torch.bfloat16:
        raise NotImplementedError(f"Trainium linear_mx supports only BF16 output, got {out_dtype}")
    if group_size != 32:
        raise NotImplementedError(
            f"Trainium linear_mx supports only group_size=32, got {group_size}"
        )
    if input_bf16.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Trainium linear_mx expects BF16 or FP16 input, got {input_bf16.dtype}")
    if weight_k_n_bf16.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"Trainium linear_mx expects BF16 or FP16 weight, got {weight_k_n_bf16.dtype}"
        )
    if input_bf16.ndim != 2 or weight_k_n_bf16.ndim != 2:
        raise ValueError("Trainium linear_mx expects input [M, K] and weight [K, N]")
    if input_bf16.shape[0] != 128:
        raise ValueError(f"Trainium linear_mx currently requires M=128, got {input_bf16.shape[0]}")
    if input_bf16.shape[1] != weight_k_n_bf16.shape[0]:
        raise ValueError(
            "Trainium linear_mx contraction mismatch: "
            f"{input_bf16.shape[1]} != {weight_k_n_bf16.shape[0]}"
        )
    if input_bf16.shape[1] % 512 != 0:
        raise ValueError(
            f"Trainium linear_mx requires K multiple of 512, got {input_bf16.shape[1]}"
        )
    if weight_k_n_bf16.shape[1] % 512 != 0:
        raise ValueError(
            f"Trainium linear_mx requires N multiple of 512, got {weight_k_n_bf16.shape[1]}"
        )
    if bias_bf16 is not None:
        if bias_bf16.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(f"Trainium linear_mx expects BF16 or FP16 bias, got {bias_bf16.dtype}")
        if bias_bf16.shape != (weight_k_n_bf16.shape[1],):
            raise ValueError(
                f"Trainium linear_mx bias must be [{weight_k_n_bf16.shape[1]}], "
                f"got {tuple(bias_bf16.shape)}"
            )


__all__ = ["dequantize_mx", "linear_mx", "matmul_mx", "quantize_mx"]
