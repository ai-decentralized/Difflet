#!/usr/bin/env python3
"""
Probe: invoke nc_matmul_sparse via the FP8-packed (float8_e4m3fn_x4) path —
the dtype NKI naturally supports for 4x compression.

Strategy:
- Weight in float8_e4m3fn (1 byte / element), shape [P=M_out, K_real]
- to_compressed_sparse with pattern (16,4) -> 4x compression along K
- View result as float8_e4m3fn_x4 (each element = 4 packed fp8 = 4 bytes)
- Build NKI kernel with x4 packed stationary + x4 packed moving (acts on same K)
- Call nc_matmul_sparse(...) inside; output to PSUM fp32 tile of size M_out x N
"""
from __future__ import annotations
import sys, traceback, time, json
from pathlib import Path
import torch
import numpy as np


def _show(name, t):
    print(f"  {name:30s} shape={tuple(t.shape)}  dtype={t.dtype}  "
          f"bytes={t.element_size() * t.numel()}")


def main() -> int:
    # ============ Step 1: build fp8 weight + 16:4 mask ============
    print("=== Step 1: build inputs ===")
    P = 128                 # partition dim (contraction, after sparse expansion)
    M_out = 64              # output rows (stationary free dim, must be <= 128)
    K_real = 512            # contraction dim, must be divisible by 16 * 4 = 64...
                            # but assertion in get_rand_mask requires divisible by 128 * R = 512
    N = 512                 # output cols (moving free dim, must be <= 512)
    L, R = 16, 4
    compress_ratio = L // R  # 4 for 16:4 pattern

    torch.manual_seed(42)
    # weight: stationary side, shape [M_out, K_real] before sparsification
    weight_fp32 = torch.randn(M_out, K_real, dtype=torch.float32) * 0.5
    weight_fp8 = weight_fp32.to(torch.float8_e4m3fn)
    _show("weight (fp8)", weight_fp8)

    # Build mask in (M, K) layout — but the sparsity API expects compression dim = 1 (along K)
    from neuronxcc.apis import sparsity
    mask = sparsity.get_rand_mask((M_out, K_real), pattern=(L, R))
    _show("mask (16:4)", mask)
    print(f"  mask nonzero fraction: {mask.float().mean().item():.4f}  (target {R/L:.4f})")

    # Apply mask in fp32 (avoid fp8 rounding clobber), then quantize back
    weight_sparse_fp32 = weight_fp32 * mask.to(weight_fp32.dtype)
    weight_sparse_fp8 = weight_sparse_fp32.to(torch.float8_e4m3fn)
    _show("weight_sparse (fp8)", weight_sparse_fp8)

    print()
    print("=== Step 2: to_compressed_sparse (fp8 input) ===")
    try:
        compressed, tag = sparsity.to_compressed_sparse(
            weight_sparse_fp8, mask, (L, R))
        _show("compressed", compressed)
        _show("tag", tag)
    except Exception as e:
        print(f"  to_compressed_sparse FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        # Fall back: maybe to_compressed_sparse doesn't accept fp8 — try fp16
        print("\n  Fallback: use fp16 weight (matches what worked earlier)")
        weight_sparse_fp16 = weight_sparse_fp32.to(torch.float16)
        compressed, tag = sparsity.to_compressed_sparse(
            weight_sparse_fp16, mask, (L, R))
        _show("compressed (fp16 path)", compressed)
        _show("tag (fp16 path)", tag)

    print()
    print("=== Step 3: also try squeeze_tags to see if it changes tag layout ===")
    try:
        squeezed = sparsity.squeeze_tags(tag, compress_ratio)
        _show("squeeze_tags(tag)", squeezed)
    except Exception as e:
        print(f"  squeeze_tags raised: {type(e).__name__}: {e}")
        squeezed = tag

    print()
    print("=== Step 4: inspect packed-dtype options ===")
    # See if torch has float8_e4m3fn_x4
    candidates = [
        'float8_e4m3fn_x4', 'float8_e5m2_x4',
        'float4_e2m1fn_x4', 'float8_e8m0fnu',
    ]
    for c in candidates:
        try:
            t = getattr(torch, c)
            print(f"  torch.{c}: {t}")
        except AttributeError:
            print(f"  torch.{c}: NOT in torch namespace")
    # In NKI:
    from neuronxcc.nki import language as nl_ncc
    print("  -- NKI nl dtypes (filter packed) --")
    for n in dir(nl_ncc):
        if not n.startswith('_') and ('x4' in n or 'e8m0' in n):
            print(f"    nl.{n}: {getattr(nl_ncc, n)}")

    print()
    print("=== Step 5: probe nc_matmul_sparse with proper x4 packed dtypes ===")
    # The kernel will accept tensors that NKI can load. Since 'compressed' came
    # back as int64, we view-reinterpret as float8_e4m3fn_x4. Element width:
    #   int64 = 8 bytes ; float8_e4m3fn_x4 = 4 bytes (4 fp8 packed)
    # so view doubles the K dim count.
    print(f"  compressed dtype before view: {compressed.dtype}  numel={compressed.numel()}  bytes={compressed.element_size()*compressed.numel()}")
    # view via uint8 then reshape into x4 (NKI's x4 dtypes don't exist in torch,
    # so we pass int32 view as a workaround — 4 bytes per element, same as x4)
    compressed_i32 = compressed.view(torch.int32).contiguous()
    _show("compressed (view int32)", compressed_i32)
    # Now compressed_i32 has shape [M_out, K_real / compress_ratio * 2] = [64, 256]
    # That's [F_stat=M_out, doubled-F] — but for NKI nc_matmul layout, we need
    # stationary[P=K_compressed_packed, F=M_out].  Transpose:
    stationary_packed = compressed_i32.T.contiguous()  # [256, 64]
    _show("stationary_packed (T)", stationary_packed)

    # moving: build as fp8 [K_real, N], pack to x4 (each elem = 4 fp8 = 4 bytes)
    moving_fp32 = torch.randn(K_real, N, dtype=torch.float32) * 0.3
    moving_fp8 = moving_fp32.to(torch.float8_e4m3fn)
    _show("moving (fp8)", moving_fp8)
    # Pack: view 4 fp8 as int32 (4 bytes), shape [K_real/4=128, N]
    moving_packed = moving_fp8.view(torch.int32).reshape(K_real // 4, N).contiguous()
    _show("moving_packed (int32 view)", moving_packed)
    # tag: rectify shape if needed; we just pass as-is and let the kernel complain
    tag_to_pass = tag.contiguous()
    _show("tag", tag_to_pass)

    # ============ build + run kernel ============
    print()
    print("=== Step 6: build NKI kernel calling nc_matmul_sparse ===")
    from neuronxcc import nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki._private.private_api import nc_matmul_sparse

    @nki.jit
    def sparse_kernel(stat_ptr, tags_ptr, moving_ptr, compress_ratio_v: int):
        # stat_ptr shape: [P=K_packed=128, F=M_out=64]
        # moving_ptr shape: [P=K_packed=128, F=N]
        # tags_ptr shape: same as stat (or smaller)
        P_in, F_stat = stat_ptr.shape
        P_m, F_mov = moving_ptr.shape
        out = nl.ndarray((F_stat, F_mov), dtype=nl.float32, buffer=nl.shared_hbm)

        stat = nl.load(stat_ptr)
        tags = nl.load(tags_ptr)
        mov = nl.load(moving_ptr)
        psum = nc_matmul_sparse(
            moving=mov, stationary=stat, tags=tags,
            compress_ratio=compress_ratio_v,
        )
        nl.store(out, value=psum)
        return out

    print("=== Step 7: baremetal run ===")
    from neuronxcc.nki import baremetal
    try:
        kn = baremetal(sparse_kernel)
        t0 = time.perf_counter()
        out = kn(stationary_packed.numpy(),
                 tag_to_pass.numpy(),
                 moving_packed.numpy(),
                 compress_ratio)
        print(f"  ✅ SPARSE CALL SUCCEEDED in {time.perf_counter()-t0:.2f}s; out shape {out.shape} dtype {out.dtype}")
        # Compare to reference dense matmul
        ref = (weight_sparse_fp32 @ moving_fp32)
        print(f"  reference dense out shape: {ref.shape}")
        cos = torch.nn.functional.cosine_similarity(
            torch.from_numpy(out).flatten().float().unsqueeze(0),
            ref.flatten().unsqueeze(0)).item()
        print(f"  cosine vs reference: {cos:.4f}")
    except Exception as e:
        print(f"  ❌ SPARSE CALL FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
