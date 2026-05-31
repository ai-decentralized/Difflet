#!/usr/bin/env python3
"""Focused minimal sparse fp8 matmul attempt — single semantic layout."""
from __future__ import annotations
import sys, time, traceback
import torch, numpy as np


def main() -> int:
    M = 64                  # output rows (stationary free dim)
    K_real = 512            # contraction dim
    N = 128                 # moving free dim
    L, R = 16, 4
    K_groups = K_real // L  # = 32 sparse-packed elements per row
    compress_ratio = 4

    print(f"M={M}, K={K_real}, N={N}, pattern=16:4, K_groups={K_groups}")

    # ---------- build sparse weight as x4-packed ----------
    g = torch.Generator().manual_seed(0)
    dense = (torch.randn(M, K_real, generator=g) * 0.4)
    gi = torch.Generator().manual_seed(1)
    kept_idx = torch.zeros(M, K_groups, R, dtype=torch.int64)
    kept_val = torch.zeros(M, K_groups, R, dtype=torch.float32)
    for i in range(M):
        for g_idx in range(K_groups):
            chosen = torch.randperm(L, generator=gi)[:R].sort().values
            kept_idx[i, g_idx] = chosen
            kept_val[i, g_idx] = dense[i, g_idx*L:(g_idx+1)*L][chosen]
    kept_fp8 = kept_val.to(torch.float8_e4m3fn)                          # [M, K_groups, R]

    # Reconstruct dense_sparse for reference matmul
    dense_sparse = torch.zeros_like(dense)
    for i in range(M):
        for g_idx in range(K_groups):
            for k in range(R):
                dense_sparse[i, g_idx*L + kept_idx[i, g_idx, k].item()] = kept_val[i, g_idx, k]

    # Pack as x4: each [R=4] fp8 -> 1 int32 (4 bytes). Per row: K_groups x4 elements.
    # Transpose to put P=K_groups in partition dim: [K_groups, M] x4 = [32, 64] int32
    stat_packed_pkm = (kept_fp8.view(torch.uint8)         # [M, K_groups, R] u8
                       .reshape(M, K_groups, R)
                       .permute(1, 0, 2)                    # [K_groups, M, R]
                       .reshape(K_groups, M*R)              # [K_g, M*R] u8
                       .view(torch.int32)                   # [K_g, M] i32 (4 u8 = 1 i32)
                       .reshape(K_groups, M)
                       .contiguous())
    print(f"  stat_packed [P=K_g, F=M]: {tuple(stat_packed_pkm.shape)} {stat_packed_pkm.dtype}")

    # Tags: unpacked uint8, [K_groups, M*R] u8 — one index per fp8 nonzero
    tags_pkm = (kept_idx.to(torch.uint8)                # [M, K_groups, R] u8
                .permute(1, 0, 2)                       # [K_groups, M, R]
                .reshape(K_groups, M*R)                 # [K_g, M*R] u8
                .contiguous())
    print(f"  tags [P=K_g, F=M*R]:     {tuple(tags_pkm.shape)} {tags_pkm.dtype}")

    # Moving: dense activation. For matmul to align with K_groups (32 sparse-packed partitions),
    # the K dim must be EXPANDED back to K_real and packed densely.
    # In x4 form: K_real / 4 = 128 dense-packed elements.
    # P partition convention: must equal K_groups=32 to match stat/tags partitions.
    # → moving needs L=16 dense fp8 K-values per partition. 16 fp8 = 16 bytes = 4 x4 elements.
    # So moving shape: [P=K_g=32, F = N * 4] x4. Each P slot holds the 16 K-values worth needed for that sparse group.
    g2 = torch.Generator().manual_seed(7)
    mov_fp8 = (torch.randn(K_real, N, generator=g2) * 0.3).to(torch.float8_e4m3fn)
    # Re-pack: per K_group, take L=16 consecutive K-values, pack as 4 contiguous x4 elements
    # mov_fp8 [K_real=512, N] -> [K_g=32, L=16, N] -> permute -> [K_g=32, N, L=16]
    # then view as int32 (4 u8 = 1 i32): [K_g, N, L/4=4] i32 -> reshape [K_g, N*4] i32
    mov_pkm = (mov_fp8.view(K_groups, L, N)              # [K_g, L, N]
               .permute(0, 2, 1)                          # [K_g, N, L]
               .contiguous()
               .view(torch.uint8)
               .reshape(K_groups, N, L)
               .view(K_groups, N * L)                     # [K_g, N*L=2048] u8
               .view(torch.int32)
               .reshape(K_groups, N * (L // 4))           # [K_g, N*4=512] i32 (each = 4 fp8 = x4)
               .contiguous())
    print(f"  mov [P=K_g, F=N*L/4]:    {tuple(mov_pkm.shape)} {mov_pkm.dtype}")

    # ---------- NKI2 kernel ----------
    from neuronxcc import nki
    import neuronxcc.nki.language as nl
    from neuronxcc.nki._private.private_api import nc_matmul_sparse
    from nki.dtype import float8_e4m3fn_x4 as FP8X4

    @nki.jit
    def sparse_kernel(stat_i32, tags_u8, mov_i32):
        # Variant: skip psum pre-alloc — use returned local_tile directly
        P_s, F_s = stat_i32.shape
        P_m, F_m = mov_i32.shape
        out_ptr = nl.ndarray((F_s, F_m), dtype=nl.float32, buffer=nl.shared_hbm)

        stat = nl.load(stat_i32).view(FP8X4)
        tags = nl.load(tags_u8)
        mov = nl.load(mov_i32).view(FP8X4)

        result = nc_matmul_sparse(
            moving=mov, stationary=stat, tags=tags, compress_ratio=compress_ratio
        )
        nl.store(out_ptr, value=result)
        return out_ptr

    from neuronxcc.nki import baremetal
    print()
    print("=== Run ===")
    try:
        kn = baremetal(sparse_kernel)
        t0 = time.perf_counter()
        out = kn(stat_packed_pkm.numpy(), tags_pkm.numpy(), mov_pkm.numpy())
        print(f"  ✅ SUCCESS in {time.perf_counter()-t0:.2f}s")
        print(f"  out shape: {out.shape}  dtype: {out.dtype}")
        print(f"  out stats: mean={out.mean():.3f} std={out.std():.3f}")
        # reference
        ref = (dense_sparse.float() @ mov_fp8.float())                # [M, N]
        print(f"  ref shape: {ref.shape}")
        print(f"  ref stats: mean={ref.mean():.3f} std={ref.std():.3f}")
        if out.shape == ref.shape:
            cos = torch.nn.functional.cosine_similarity(
                torch.from_numpy(out).flatten().unsqueeze(0),
                ref.flatten().unsqueeze(0)
            ).item()
            print(f"  ✨ cosine vs reference: {cos:.4f}")
        return 0
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
