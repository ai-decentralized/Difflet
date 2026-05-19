"""Torch-native MX reference implementation."""

from __future__ import annotations

import torch

SUPPORTED_DTYPE = "float8_e4m3fn_x4"
GROUP_SIZE = 32
_FLOAT32_EXP_BIAS = 127

# Per-MXFP8-format parameters: (max binade exponent, max representable
# magnitude, torch fp8 dtype). Mirrors the canonical table in
# ``nkilib.core.utils.mx_torch_common.quantize_to_mx`` so the CPU
# reference and the Trainium kernels agree bit-for-bit on the scale law.
_MX_PARAMS = {
    "float8_e4m3fn_x4": (8, 448.0, torch.float8_e4m3fn),
    "float8_e5m2_x4": (15, 57344.0, torch.float8_e5m2),
}
SUPPORTED_DTYPES = frozenset(_MX_PARAMS)


def _validate_common(dtype: str, group_size: int) -> None:
    if dtype not in _MX_PARAMS:
        raise NotImplementedError(
            f"CPU MX reference supports {sorted(_MX_PARAMS)!r}; got {dtype!r}"
        )
    if group_size != GROUP_SIZE:
        raise NotImplementedError(
            f"CPU MX reference supports only group_size={GROUP_SIZE}; got {group_size}"
        )


def _validate_quantize_input(x: torch.Tensor) -> None:
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"quantize_mx expects BF16 or FP16 input, got {x.dtype}")
    if x.ndim < 2:
        raise ValueError(f"quantize_mx expects at least 2 dimensions, got {x.ndim}")
    if x.shape[-2] % 8 != 0 or x.shape[-1] % 4 != 0:
        raise ValueError(
            "quantize_mx requires the final dimensions to be multiples of "
            f"(8, 4), got {tuple(x.shape[-2:])}"
        )


def _get_ieee_frexp(tensor: torch.Tensor) -> torch.Tensor:
    int_view = tensor.to(torch.float32).view(torch.int32)
    exp_bits = (int_view >> 23) & 0xFF
    exp = exp_bits - _FLOAT32_EXP_BIAS
    return torch.where(tensor == 0.0, torch.full_like(exp, -126), exp)


def quantize_mx(
    x: torch.Tensor,
    *,
    dtype: str = SUPPORTED_DTYPE,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP16 input to MXFP8-E4M3FN packed as ``uint32`` x4."""

    _validate_common(dtype, group_size)
    _validate_quantize_input(x)
    max_exp, max_val, fp8_dtype = _MX_PARAMS[dtype]

    *prefix, p_dim, f_dim = x.shape
    sp_dim = p_dim // 8
    sf_dim = f_dim // 4

    exp = _get_ieee_frexp(x)
    exp_blocks = exp.reshape(*prefix, sp_dim, 8, sf_dim, 4)
    max_exp_per_block = torch.amax(exp_blocks, dim=(-3, -1))
    scale = (max_exp_per_block + _FLOAT32_EXP_BIAS - max_exp).to(torch.uint8)

    scale_exp = scale.to(torch.int32) - _FLOAT32_EXP_BIAS
    scale_blocks = torch.pow(2.0, scale_exp.float())
    expanded_scale = (
        scale_blocks.unsqueeze(-2)
        .unsqueeze(-1)
        .expand(*prefix, sp_dim, 8, sf_dim, 4)
        .contiguous()
        .reshape(*prefix, p_dim, f_dim)
    )

    mx_data = torch.clamp(x / expanded_scale, -max_val, max_val)
    mx_data = mx_data.to(fp8_dtype).contiguous()
    packed = mx_data.view(torch.uint32).reshape(*prefix, p_dim, sf_dim)
    return packed, scale.contiguous()


def dequantize_mx(
    data: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: str = SUPPORTED_DTYPE,
    output_dtype: torch.dtype = torch.bfloat16,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Dequantize MXFP8-E4M3FN ``uint32`` x4 data using E8M0 scales."""

    _validate_common(dtype, group_size)
    if data.dtype != torch.uint32:
        raise TypeError(f"dequantize_mx expects uint32 packed data, got {data.dtype}")
    if scale.dtype != torch.uint8:
        raise TypeError(f"dequantize_mx expects uint8 scale, got {scale.dtype}")
    if data.ndim < 2:
        raise ValueError(f"dequantize_mx expects at least 2 dimensions, got {data.ndim}")
    if scale.shape != (*data.shape[:-2], data.shape[-2] // 8, data.shape[-1]):
        raise ValueError(
            "scale shape must be data.shape[:-2] + "
            f"(data.shape[-2] // 8, data.shape[-1]); got data={tuple(data.shape)} "
            f"scale={tuple(scale.shape)}"
        )

    _, _, fp8_dtype = _MX_PARAMS[dtype]
    *prefix, p_dim, sf_dim = data.shape
    float8_data = data.contiguous().reshape(-1).view(fp8_dtype)
    float8_data = float8_data.reshape(*prefix, p_dim, sf_dim * 4)

    scale_exp = scale.to(torch.int32) - _FLOAT32_EXP_BIAS
    scale_blocks = torch.pow(2.0, scale_exp.float())
    expanded_scale = (
        scale_blocks.unsqueeze(-2)
        .unsqueeze(-1)
        .expand(*prefix, p_dim // 8, 8, sf_dim, 4)
        .contiguous()
        .reshape(*prefix, p_dim, sf_dim * 4)
    )
    return (float8_data.to(torch.float32) * expanded_scale).to(output_dtype)


def matmul_mx(
    a_mx: torch.Tensor,
    a_scale: torch.Tensor,
    b_mx: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    dtype: str = SUPPORTED_DTYPE,
    out_dtype: torch.dtype = torch.bfloat16,
    accumulate: bool = False,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Reference ``A @ B`` implemented as dequantize plus ``torch.matmul``."""

    del accumulate
    a = dequantize_mx(
        a_mx,
        a_scale,
        dtype=dtype,
        output_dtype=torch.float32,
        group_size=group_size,
    )
    b = dequantize_mx(
        b_mx,
        b_scale,
        dtype=dtype,
        output_dtype=torch.float32,
        group_size=group_size,
    )
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"matmul_mx contraction mismatch: {a.shape[-1]} != {b.shape[-2]}"
        )
    return torch.matmul(a, b).to(out_dtype)


def dequantize_mx_hardware_tile(
    data: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: str = SUPPORTED_DTYPE,
    output_dtype: torch.dtype = torch.float32,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Dequantize an NCv4 MX tile to ``[K_packed, free, 4]``."""

    _validate_common(dtype, group_size)
    if data.dtype != torch.uint32:
        raise TypeError(f"expected uint32 packed MX data, got {data.dtype}")
    if scale.dtype != torch.uint8:
        raise TypeError(f"expected uint8 MX scale, got {scale.dtype}")
    if data.ndim != 2:
        raise ValueError(f"expected 2D hardware tile data, got {data.ndim}D")
    if scale.shape != (data.shape[0] // 8, data.shape[1]):
        raise ValueError(
            f"scale shape must be ({data.shape[0] // 8}, {data.shape[1]}), got {tuple(scale.shape)}"
        )

    _, _, fp8_dtype = _MX_PARAMS[dtype]
    k_packed, free_dim = data.shape
    unpacked = data.contiguous().reshape(-1).view(fp8_dtype)
    unpacked = unpacked.reshape(k_packed, free_dim, 4).to(torch.float32)

    scale_exp = scale.to(torch.int32) - _FLOAT32_EXP_BIAS
    scale_blocks = torch.pow(2.0, scale_exp.float())
    expanded_scale = (
        scale_blocks.unsqueeze(1)
        .unsqueeze(-1)
        .expand(k_packed // 8, 8, free_dim, 4)
        .contiguous()
        .reshape(k_packed, free_dim, 4)
    )
    return (unpacked * expanded_scale).to(output_dtype)


def matmul_mx_single_tile_reference(
    stationary_mx: torch.Tensor,
    stationary_scale: torch.Tensor,
    moving_mx: torch.Tensor,
    moving_scale: torch.Tensor,
    *,
    dtype: str = SUPPORTED_DTYPE,
    out_dtype: torch.dtype = torch.bfloat16,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Reference for one ``nc_matmul_mx`` tile in hardware layout."""

    stationary = dequantize_mx_hardware_tile(
        stationary_mx,
        stationary_scale,
        dtype=dtype,
        output_dtype=torch.float32,
        group_size=group_size,
    )
    moving = dequantize_mx_hardware_tile(
        moving_mx,
        moving_scale,
        dtype=dtype,
        output_dtype=torch.float32,
        group_size=group_size,
    )
    return matmul_mx_single_tile_reference_from_dequantized(
        stationary,
        moving,
        out_dtype=out_dtype,
    )


def matmul_mx_single_tile_reference_from_dequantized(
    stationary: torch.Tensor,
    moving: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference one hardware tile from dequantized ``[K, F, 4]`` tensors."""

    if stationary.shape[0] != moving.shape[0]:
        raise ValueError(
            f"hardware tile contraction mismatch: {stationary.shape[0]} "
            f"!= {moving.shape[0]}"
        )
    return torch.einsum("kmq,knq->mn", stationary, moving).to(out_dtype)


def matmul_mx_k_tiles_reference(
    stationary_mx: torch.Tensor,
    stationary_scale: torch.Tensor,
    moving_mx: torch.Tensor,
    moving_scale: torch.Tensor,
    *,
    dtype: str = SUPPORTED_DTYPE,
    out_dtype: torch.dtype = torch.bfloat16,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Reference K-axis multi-tile ``nc_matmul_mx`` accumulation."""

    _validate_common(dtype, group_size)
    if stationary_mx.ndim != 3 or moving_mx.ndim != 3:
        raise ValueError("K-tile reference expects 3D packed data tensors")
    if stationary_mx.shape[0] != moving_mx.shape[0]:
        raise ValueError(
            f"K_tiles mismatch: {stationary_mx.shape[0]} != {moving_mx.shape[0]}"
        )
    if stationary_scale.shape[0] != stationary_mx.shape[0]:
        raise ValueError("stationary scale K_tiles must match stationary data")
    if moving_scale.shape[0] != moving_mx.shape[0]:
        raise ValueError("moving scale K_tiles must match moving data")

    out = torch.zeros(
        (stationary_mx.shape[2], moving_mx.shape[2]),
        dtype=torch.float32,
    )
    for k_idx in range(stationary_mx.shape[0]):
        out += matmul_mx_single_tile_reference(
            stationary_mx[k_idx],
            stationary_scale[k_idx],
            moving_mx[k_idx],
            moving_scale[k_idx],
            dtype=dtype,
            out_dtype=torch.float32,
            group_size=group_size,
        )
    return out.to(out_dtype)


def _validate_linear_mx_inputs(
    input_bf16: torch.Tensor,
    weight_k_n_bf16: torch.Tensor,
    bias_bf16: torch.Tensor | None,
) -> None:
    if input_bf16.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"linear_mx input must be BF16 or FP16, got {input_bf16.dtype}")
    if weight_k_n_bf16.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"linear_mx weight must be BF16 or FP16, got {weight_k_n_bf16.dtype}"
        )
    if input_bf16.ndim != 2:
        raise ValueError(f"linear_mx input must be 2D [M, K], got {input_bf16.ndim}D")
    if weight_k_n_bf16.ndim != 2:
        raise ValueError(
            f"linear_mx weight must be 2D [K, N], got {weight_k_n_bf16.ndim}D"
        )
    if input_bf16.shape[0] != 128:
        raise ValueError(f"linear_mx currently requires M=128, got {input_bf16.shape[0]}")
    if input_bf16.shape[1] != weight_k_n_bf16.shape[0]:
        raise ValueError(
            "linear_mx contraction mismatch: "
            f"{input_bf16.shape[1]} != {weight_k_n_bf16.shape[0]}"
        )
    if input_bf16.shape[1] % 512 != 0:
        raise ValueError(f"linear_mx requires K multiple of 512, got {input_bf16.shape[1]}")
    if weight_k_n_bf16.shape[1] != 512:
        raise ValueError(f"linear_mx currently requires N=512, got {weight_k_n_bf16.shape[1]}")
    if bias_bf16 is not None:
        if bias_bf16.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(f"linear_mx bias must be BF16 or FP16, got {bias_bf16.dtype}")
        if bias_bf16.shape != (weight_k_n_bf16.shape[1],):
            raise ValueError(
                f"linear_mx bias must be [{weight_k_n_bf16.shape[1]}], "
                f"got {tuple(bias_bf16.shape)}"
            )


def pack_linear_mx_inputs(
    input_bf16: torch.Tensor,
    weight_k_n_bf16: torch.Tensor,
    bias_bf16: torch.Tensor | None = None,
    *,
    dtype: str = SUPPORTED_DTYPE,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack logical ``input @ weight`` tensors into MX K-tile kernel inputs."""

    _validate_common(dtype, group_size)
    _validate_linear_mx_inputs(input_bf16, weight_k_n_bf16, bias_bf16)

    k_tiles = input_bf16.shape[1] // 512
    stationary_tiles = []
    moving_tiles = []
    for tile_idx in range(k_tiles):
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
        stationary_tiles.append(activation_native)
        moving_tiles.append(weight_native)

    stationary = torch.stack(stationary_tiles, dim=0)
    moving = torch.stack(moving_tiles, dim=0)
    stationary_mx, stationary_scale = quantize_mx(
        stationary,
        dtype=dtype,
        group_size=group_size,
    )
    moving_mx, moving_scale = quantize_mx(
        moving,
        dtype=dtype,
        group_size=group_size,
    )
    return stationary_mx, stationary_scale, moving_mx, moving_scale


def linear_mx_reference(
    input_bf16: torch.Tensor,
    weight_k_n_bf16: torch.Tensor,
    bias_bf16: torch.Tensor | None = None,
    *,
    dtype: str = SUPPORTED_DTYPE,
    out_dtype: torch.dtype = torch.bfloat16,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Reference logical Linear using MX K-tile packing and accumulation."""

    packed = pack_linear_mx_inputs(
        input_bf16,
        weight_k_n_bf16,
        bias_bf16,
        dtype=dtype,
        group_size=group_size,
    )
    out = matmul_mx_k_tiles_reference(
        *packed,
        dtype=dtype,
        out_dtype=torch.float32,
        group_size=group_size,
    )
    if bias_bf16 is not None:
        out = out + bias_bf16.to(torch.float32)
    return out.to(out_dtype)


def linear_mx(
    input_bf16: torch.Tensor,
    weight_k_n_bf16: torch.Tensor,
    bias_bf16: torch.Tensor | None = None,
    *,
    dtype: str = SUPPORTED_DTYPE,
    out_dtype: torch.dtype = torch.bfloat16,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    return linear_mx_outer_n_reference(
        input_bf16,
        weight_k_n_bf16,
        bias_bf16,
        dtype=dtype,
        out_dtype=out_dtype,
        group_size=group_size,
    )


def linear_mx_outer_n_reference(
    input_bf16: torch.Tensor,
    weight_k_n_bf16: torch.Tensor,
    bias_bf16: torch.Tensor | None = None,
    *,
    dtype: str = SUPPORTED_DTYPE,
    out_dtype: torch.dtype = torch.bfloat16,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Reference logical Linear with host-side outer-N composition."""

    if weight_k_n_bf16.ndim != 2:
        raise ValueError(
            f"linear_mx weight must be 2D [K, N], got {weight_k_n_bf16.ndim}D"
        )
    if weight_k_n_bf16.shape[1] % 512 != 0:
        raise ValueError(
            f"linear_mx outer-N requires N multiple of 512, got {weight_k_n_bf16.shape[1]}"
        )
    if bias_bf16 is not None and bias_bf16.shape != (weight_k_n_bf16.shape[1],):
        raise ValueError(
            f"linear_mx bias must be [{weight_k_n_bf16.shape[1]}], "
            f"got {tuple(bias_bf16.shape)}"
        )

    outputs = []
    for n_start in range(0, weight_k_n_bf16.shape[1], 512):
        n_end = n_start + 512
        bias_tile = bias_bf16[n_start:n_end] if bias_bf16 is not None else None
        outputs.append(
            linear_mx_reference(
                input_bf16,
                weight_k_n_bf16[:, n_start:n_end].contiguous(),
                bias_tile.contiguous() if bias_tile is not None else None,
                dtype=dtype,
                out_dtype=out_dtype,
                group_size=group_size,
            )
        )
    return torch.cat(outputs, dim=1)


__all__ = [
    "SUPPORTED_DTYPES",
    "dequantize_mx",
    "dequantize_mx_hardware_tile",
    "linear_mx",
    "linear_mx_outer_n_reference",
    "linear_mx_reference",
    "matmul_mx",
    "matmul_mx_k_tiles_reference",
    "matmul_mx_single_tile_reference",
    "matmul_mx_single_tile_reference_from_dequantized",
    "pack_linear_mx_inputs",
    "quantize_mx",
]
