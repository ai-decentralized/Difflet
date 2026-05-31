#!/usr/bin/env python3
"""
Probe: can we actually invoke nc_matmul_sparse (NKI2 private API) on Trainium?

If yes:
  - Hardware sparse matmul path is reachable TODAY (not in NKI3, but in NKI2 private)
  - We can measure real 4x speedup on 16:4 pattern
  - Gives strong leverage to push AWS for NKI3 exposure

Pipeline:
  1. Build BF16 weight + 16:4 mask via neuronxcc.apis.sparsity.get_rand_mask
  2. Call to_compressed_sparse -> (compressed, tag) on HOST
  3. Move (compressed, tag, moving) to Trainium
  4. Call nki.baremetal(nc_matmul_sparse) kernel
  5. Compare timing vs dense nc_matmul of equivalent dense weight
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import numpy as np


def _build_dense_matmul_kernel():
    """Standard NKI2 dense nc_matmul for baseline."""
    from neuronxcc import nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa

    @nki.jit
    def dense_matmul_kernel(stationary_ptr, moving_ptr):
        # stationary: [P=128, K]
        # moving:     [P=128, N]
        # out:        [N, K]  (or [K, N], depending on conv)
        P, K = stationary_ptr.shape
        P_m, N = moving_ptr.shape
        out_ptr = nl.ndarray((K, N), dtype=nl.float32, buffer=nl.shared_hbm)

        stationary_sbuf = nl.load(stationary_ptr)
        moving_sbuf = nl.load(moving_ptr)
        psum = nl.zeros((K, N), dtype=nl.float32, buffer=nl.psum)
        psum[...] = nisa.nc_matmul(stationary_sbuf, moving_sbuf)
        nl.store(out_ptr, value=psum)
        return out_ptr

    return dense_matmul_kernel


def _build_sparse_matmul_kernel():
    """NKI2 sparse nc_matmul_sparse via private API."""
    from neuronxcc import nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki._private.private_api import nc_matmul_sparse

    @nki.jit
    def sparse_matmul_kernel(stationary_compressed_ptr, tags_ptr, moving_ptr, compress_ratio: int):
        # stationary_compressed: [P=128, K_c=K/compress_ratio]
        # tags:                  [P=128, K_c]
        # moving:                [P=128, N]
        # out:                   [K=K_c*compress_ratio, N]
        P, K_c = stationary_compressed_ptr.shape
        P_m, N = moving_ptr.shape
        K = K_c * compress_ratio
        out_ptr = nl.ndarray((K, N), dtype=nl.float32, buffer=nl.shared_hbm)

        stationary_sbuf = nl.load(stationary_compressed_ptr)
        tags_sbuf = nl.load(tags_ptr)
        moving_sbuf = nl.load(moving_ptr)
        psum = nc_matmul_sparse(
            moving=moving_sbuf,
            stationary=stationary_sbuf,
            tags=tags_sbuf,
            compress_ratio=compress_ratio,
        )
        nl.store(out_ptr, value=psum)
        return out_ptr

    return sparse_matmul_kernel


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--P", type=int, default=128)
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--pattern", choices=("16:4", "4:8"), default="16:4")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("/tmp/nova_sparse_matmul_nki2_probe.json"))
    p.add_argument("--mode", choices=("simulator", "baremetal"), default="baremetal")
    args = p.parse_args()

    L, R = (int(x) for x in args.pattern.split(":"))
    compress_ratio = L // R  # 16:4 -> compress 4x ; 4:8 means 4 of 8 -> compress 2x
    assert args.K % (R * 128) == 0, f"K={args.K} must be divisible by R*128={R*128}"

    print(f"=== Setup ===")
    print(f"shape: P={args.P}, K={args.K}, N={args.N}, pattern={args.pattern} -> compress_ratio={compress_ratio}")

    # Build weight + mask — use float16 throughout so numpy can ingest
    from neuronxcc.apis import sparsity
    torch.manual_seed(42)
    weight = (torch.randn(args.P, args.K, dtype=torch.float32) + 1.0).to(torch.float16)
    mask = sparsity.get_rand_mask((args.P, args.K), pattern=(L, R))
    weight_sparse = (weight * mask.to(weight.dtype)).contiguous()
    compressed, tag = sparsity.to_compressed_sparse(weight_sparse, mask, (L, R))
    print(f"  weight dense:      {weight.shape} {weight.dtype}")
    print(f"  weight sparse:     {weight_sparse.shape} {weight_sparse.dtype} (nonzero {1-(weight_sparse==0).float().mean().item():.3f})")
    print(f"  compressed:        {compressed.shape} {compressed.dtype}")
    print(f"  tag:               {tag.shape} {tag.dtype}")
    moving = (torch.randn(args.P, args.N, dtype=torch.float32) + 0.5).to(torch.float16)
    print(f"  moving:            {moving.shape} {moving.dtype}")

    # Build kernels
    print(f"\n=== Build kernels ===")
    try:
        dense_kernel = _build_dense_matmul_kernel()
        print("  dense kernel built")
        sparse_kernel = _build_sparse_matmul_kernel()
        print("  sparse kernel built")
    except Exception as e:
        print(f"  FAILED to build kernels: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2

    # === try simulator first ===
    if args.mode == "simulator":
        print(f"\n=== Try simulator ===")
        try:
            from neuronxcc.nki import simulate_kernel
        except ImportError:
            try:
                from nki.simulator import simulate_kernel
            except ImportError:
                from neuronxcc.nki._private.test import simulate_kernel
        try:
            t0 = time.perf_counter()
            out_dense = simulate_kernel(dense_kernel, (weight.numpy(), moving.numpy()))
            print(f"  dense simulator out shape: {out_dense.shape}, took {time.perf_counter()-t0:.2f}s")
        except Exception as e:
            print(f"  dense sim FAIL: {type(e).__name__}: {e}")

        try:
            t0 = time.perf_counter()
            out_sparse = simulate_kernel(
                sparse_kernel,
                (compressed.numpy(), tag.numpy(), moving.numpy()),
                {"compress_ratio": compress_ratio},
            )
            print(f"  sparse simulator out shape: {out_sparse.shape}, took {time.perf_counter()-t0:.2f}s")
        except Exception as e:
            print(f"  sparse sim FAIL: {type(e).__name__}: {e}")
        return 0

    # === baremetal mode ===
    print(f"\n=== Compile + run on Trainium (baremetal) ===")
    print("  using neuronxcc.nki.baremetal()")
    try:
        from neuronxcc.nki import baremetal
    except ImportError as e:
        print(f"  baremetal import failed: {e}")
        # fallback: try to compile via XLA
        try:
            import torch_xla.core.xla_model as xm
        except ImportError:
            print("  torch_xla also missing — cannot run on hardware")
            return 3
        device = xm.xla_device()
        print(f"  fallback: torch_xla device {device}")
        # ... defer to a different runtime; this probe focuses on showing whether the kernel
        # even compiles. simulator path above should already show that.
        return 4

    # baremetal: returns numpy result and saves NEFF
    print(f"\n=== Compile sparse kernel + try a single call ===")
    # nl.load doesn't accept int64 — view-reinterpret as int32 (same bits, supported dtype)
    compressed_i32 = compressed.view(torch.int32).contiguous()
    print(f"  compressed (viewed as int32): {compressed_i32.shape} {compressed_i32.dtype}")
    try:
        sparse_baremetal = baremetal(sparse_kernel)
        t0 = time.perf_counter()
        out = sparse_baremetal(compressed_i32.numpy(), tag.numpy(), moving.numpy(), compress_ratio)
        print(f"  SPARSE CALL SUCCEEDED in {time.perf_counter()-t0:.2f}s; out shape {out.shape}")
    except Exception as e:
        print(f"  SPARSE CALL FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        # don't return — also try dense baseline so we know how far we got
        out = None

    print(f"\n=== Compile dense kernel + single call (baseline) ===")
    try:
        dense_baremetal = baremetal(dense_kernel)
        t0 = time.perf_counter()
        out_d = dense_baremetal(weight.numpy(), moving.numpy())
        print(f"  DENSE CALL SUCCEEDED in {time.perf_counter()-t0:.2f}s; out shape {out_d.shape}")
    except Exception as e:
        print(f"  DENSE CALL FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
