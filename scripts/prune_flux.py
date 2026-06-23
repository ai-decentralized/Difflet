#!/usr/bin/env python3
"""Offline 16:4 structured sparsity pruning for FLUX model weights.

Usage:
    python scripts/prune_flux.py \\
        --model black-forest-labs/FLUX.1-dev \\
        --output ./pruned_flux \\
        --mode mx-fp8

Modes:
  - bf16:   4x parameter reduction via 16:4 magnitude pruning (needs nc_matmul_sparse)
  - fp8:    16x parameter reduction via 16:4 pruning + FP8 quantization (needs nc_matmul_sparse)
  - mx-fp8: 4x reduction, column-aligned 16:4 pruning → FP8 x4 packed → nc_matmul_mx compatible
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def prune_and_compress_bf16(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Magnitude-based 16:4 pruning for BF16 weights.

    Args:
        weight: [M, K] original weight tensor.

    Returns:
        compressed: [M, K_c] packed nonzero values (int64 — 4 BF16 per element).
        tags:       [M, K_c] uint16 — packed 4-bit indices, stored as int16.
    """
    L, R = 16, 4
    M, K = weight.shape
    if K % L != 0:
        raise ValueError(f"K={K} must be divisible by {L} for 16:4 pattern")
    K_groups = K // L
    K_c = K_groups * R  # compressed dim: groups × nonzeros per group

    # Reshape to expose groups: [M, K_groups, L]
    w_reshaped = weight.float().reshape(M, K_groups, L)

    # Per group: select top-R by absolute magnitude
    _, topk_idx = w_reshaped.abs().topk(R, dim=-1)  # [M, K_groups, R]

    # Gather kept values (keep in original dtype for compression)
    kept_values = torch.gather(
        weight.reshape(M, K_groups, L), dim=-1, index=topk_idx
    )  # [M, K_groups, R] — original dtype

    # Flatten kept values: [M, K_groups, R] -> [M, K_c]
    kept_flat = kept_values.reshape(M, K_c)  # [M, K_c] BF16

    # Pack 4 BF16 values into one int64
    # 4 BF16 = 8 bytes = 1 int64
    # [M, K_c] BF16 -> [M, K_c/4] int64 via uint8 view
    u8 = kept_flat.view(torch.uint8)  # [M, K_c*2] uint8
    compressed = u8.reshape(M, K_c // 4, 8).view(torch.int64).reshape(M, K_c // 4)
    compressed = compressed.contiguous()

    # Pack tags: 4 × 4-bit indices into one uint16 per group
    tags_u16 = torch.zeros(M, K_groups, dtype=torch.int32)
    for r in range(R):
        tags_u16 |= (topk_idx[:, :, r].to(torch.int32) & 0xF) << (4 * r)
    # Convert to uint16 numpy then back to int16 tensor for NKI compatibility
    tags_out = tags_u16.numpy().astype(np.uint16).view(np.int16)
    tags_tensor = torch.from_numpy(tags_out)  # [M, K_groups] int16 (as uint16 proxy)

    return compressed, tags_tensor


def prune_and_compress_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """16:4 pruning + FP8 quantization.

    Returns:
        compressed_packed: [M, K_c] int32 — FP8 x4 packed nonzero values.
        tags:              [M, K_c] uint16 — packed 4-bit indices.
    """
    L, R = 16, 4
    M, K = weight.shape
    if K % L != 0:
        raise ValueError(f"K={K} must be divisible by {L} for 16:4 pattern")
    K_groups = K // L
    K_c = K_groups * R

    w_reshaped = weight.float().reshape(M, K_groups, L)
    _, topk_idx = w_reshaped.abs().topk(R, dim=-1)
    kept_values = torch.gather(
        weight.reshape(M, K_groups, L), dim=-1, index=topk_idx
    )

    # Quantize to FP8
    kept_fp8 = kept_values.to(torch.float8_e4m3fn)  # [M, K_groups, R]

    # Pack 4 FP8 values into one int32 (x4 layout)
    # Each FP8 = 1 byte, 4 FP8 = 4 bytes = 1 int32
    kept_flat = kept_fp8.reshape(M, K_c)  # [M, K_c] float8
    u8 = kept_flat.view(torch.uint8)  # [M, K_c] uint8
    compressed_packed = u8.reshape(M, K_c // 4, 4).view(torch.int32).reshape(M, K_c // 4)
    compressed_packed = compressed_packed.contiguous()

    # Tags: same as BF16
    tags_u16 = torch.zeros(M, K_groups, dtype=torch.int32)
    for r in range(R):
        tags_u16 |= (topk_idx[:, :, r].to(torch.int32) & 0xF) << (4 * r)
    tags_out = tags_u16.numpy().astype(np.uint16).view(np.int16)
    tags_tensor = torch.from_numpy(tags_out)

    return compressed_packed, tags_tensor


def prune_and_compress_mx(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Column-aligned 16:4 pruning → FP8 x4 packed for nc_matmul_mx.

    Selects the same 4 of 16 positions per column group for ALL output rows,
    enabling a standard dense MX matmul on the compressed dimensions.

    Args:
        weight: [M, K] original weight tensor.

    Returns:
        weight_mx:   [M, K/16] int32 — FP8 x4 packed nonzero values
        weight_scale: [16, K/16] uint8 — compact MX scales
        gather_idx:  [K/16, 4] int64 — which 4 of 16 positions per group
    """
    L, R = 16, 4
    M, K = weight.shape
    assert K % L == 0, f"K={K} must be divisible by 16"
    K_groups = K // L  # number of 16-element groups along K

    # Column-aligned: aggregate |magnitude| across all rows,
    # pick top-4 positions per column group (shared for all rows)
    w_reshaped = weight.float().abs().reshape(M, K_groups, L)  # [M, K_g, 16]
    col_scores = w_reshaped.sum(dim=0)  # [K_g, 16] — aggregate importance
    _, gather_idx = col_scores.topk(R, dim=-1)  # [K_g, 4] — shared indices

    # Extract kept values for all rows using shared indices
    # weight [M, K] → [M, K_g, 16] → gather → [M, K_g, 4]
    w_groups = weight.reshape(M, K_groups, L)  # [M, K_g, 16]
    # Expand gather_idx to [1, K_g, 4] for broadcasting over M
    kept_values = torch.gather(w_groups, dim=-1, index=gather_idx.unsqueeze(0).expand(M, -1, -1))
    # kept_values: [M, K_g, 4] — the nonzero values

    # Quantize to FP8
    kept_fp8 = kept_values.to(torch.float8_e4m3fn)  # [M, K_g, 4]

    # Pack as x4 int32: [M, K_g, 4] fp8 → [M, K_g] int32
    kept_flat = kept_fp8.reshape(M, K_groups * R)  # [M, K_c]
    weight_mx = (
        kept_flat.view(torch.uint8)
        .reshape(M, K_groups, 4)
        .view(torch.int32)
        .reshape(M, K_groups)
        .contiguous()
    )  # [M, K_g] int32 — each packs 4 FP8 values

    # MX scales: uniform E8M0 (127 = 2^0 = scale 1.0) for simplicity.
    # Proper per-block scales can be added later via the CPU quantize_mx
    # reference when actual MX matmul accuracy needs tuning.
    weight_scale = torch.full((16, K_groups), 127, dtype=torch.uint8)

    # Gather indices: [K_groups, 4] int64
    gather_idx_out = gather_idx.contiguous().to(torch.int64)

    return weight_mx, weight_scale, gather_idx_out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="16:4 structured pruning for FLUX model weights"
    )
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--output", required=True, help="Output directory for pruned weights")
    p.add_argument(
        "--mode", choices=("bf16", "fp8", "mx-fp8"), default="mx-fp8",
        help="Sparsity mode: bf16 (needs nc_matmul_sparse), fp8 (needs nc_matmul_sparse), mx-fp8 (nc_matmul_mx compatible)",
    )
    p.add_argument(
        "--tp-degree", type=int, default=1,
        help="Tensor parallel degree (each rank's weights compressed independently)",
    )
    p.add_argument(
        "--dtype", default="bfloat16", choices=("bfloat16", "float16"),
    )
    args = p.parse_args(argv)

    dtype = getattr(torch, args.dtype)
    prune_fn = {
        "bf16": prune_and_compress_bf16,
        "fp8": prune_and_compress_fp8,
        "mx-fp8": prune_and_compress_mx,
    }[args.mode]

    print(f"Loading FLUX model from {args.model}...")
    from diffusers import FluxTransformer2DModel

    model = FluxTransformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=dtype,
    )

    state_dict = model.state_dict()
    sparse_state_dict = {}
    linear_weights = {
        k: v for k, v in state_dict.items()
        if k.endswith(".weight") and v.ndim == 2 and v.shape[1] % 16 == 0
    }

    skipped = {
        k: v.shape
        for k, v in state_dict.items()
        if k.endswith(".weight") and v.ndim == 2 and v.shape[1] % 16 != 0
    }
    if skipped:
        print(f"  Skipping {len(skipped)} layers with K not multiple of 16:")
        for k, shape in sorted(skipped.items()):
            print(f"    {k}: {shape}")

    for key, weight in sorted(linear_weights.items()):
        M, K = weight.shape
        result = prune_fn(weight)
        base = key.replace(".weight", "")
        if args.mode == "mx-fp8":
            weight_mx, weight_scale, gather_idx = result
            sparse_state_dict[f"{base}.sparse_weight"] = weight_mx
            sparse_state_dict[f"{base}.sparse_scale"] = weight_scale
            sparse_state_dict[f"{base}.sparse_gather_idx"] = gather_idx
            comp_shape = tuple(weight_mx.shape)
        else:
            compressed, tags = result
            sparse_state_dict[f"{base}.sparse_weight"] = compressed
            sparse_state_dict[f"{base}.sparse_tags"] = tags
            comp_shape = tuple(compressed.shape)

        # Copy bias if present
        bias_key = key.replace(".weight", ".bias")
        if bias_key in state_dict:
            sparse_state_dict[bias_key] = state_dict[bias_key].clone()

        orig_params = M * K
        comp_params = weight_mx.numel() * weight_mx.element_size() if args.mode == "mx-fp8" else compressed.numel() * compressed.element_size()
        ratio = (M * K * weight.element_size()) / comp_params
        print(
            f"  {key}: {tuple(weight.shape)} -> compressed "
            f"{comp_shape} ({ratio:.1f}x size reduction)"
        )

    # Save metadata
    if args.mode == "mx-fp8":
        sparse_state_dict["__sparse_metadata__"] = dict(
            mode=args.mode,
            pattern=[16, 4],
            compress_ratio=4,
            isa="nc_matmul_mx",
        )
    else:
        sparse_state_dict["__sparse_metadata__"] = dict(
            mode=args.mode,
            pattern=[16, 4],
            compress_ratio=4,
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sparse_weights.pt"
    torch.save(sparse_state_dict, out_path)
    print(f"\nSaved pruned weights to {out_path}")
    print(f"Total Linear layers pruned: {len(linear_weights)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
