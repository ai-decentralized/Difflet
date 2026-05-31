#!/usr/bin/env python3
"""
Pivot: use NKI3 surface (top-level nki) + import nc_matmul_sparse from NKI2 private.
NKI3 has float8_e4m3fn_x4 native + better SBUF alignment story.
"""
from __future__ import annotations
import sys, time, traceback, numpy as np
import torch


def _show(name, t):
    nb = t.element_size() * t.numel() if hasattr(t, 'element_size') else t.nbytes
    print(f"  {name:30s} shape={tuple(t.shape)}  dtype={t.dtype}  bytes={nb}")


def build_sparse_fp8(M: int, K_real: int, seed: int = 0):
    L, R = 16, 4
    K_groups = K_real // L
    g = torch.Generator().manual_seed(seed)
    dense = (torch.randn(M, K_real, generator=g, dtype=torch.float32) * 0.4)

    g_idx = torch.Generator().manual_seed(seed + 1)
    kept_values = torch.zeros(M, K_groups, R, dtype=torch.float32)
    kept_indices = torch.zeros(M, K_groups, R, dtype=torch.int64)
    for i in range(M):
        for gi in range(K_groups):
            perm = torch.randperm(L, generator=g_idx)
            chosen = perm[:R].sort().values
            kept_indices[i, gi] = chosen
            kept_values[i, gi] = dense[i, gi*L:(gi+1)*L][chosen]

    dense_sparse = torch.zeros_like(dense)
    for i in range(M):
        for gi in range(K_groups):
            for k in range(R):
                pos = kept_indices[i, gi, k].item()
                dense_sparse[i, gi*L+pos] = kept_values[i, gi, k]

    kept_fp8 = kept_values.to(torch.float8_e4m3fn)                        # [M, K_g, R]
    stat_packed_i32 = (kept_fp8.view(torch.uint8).reshape(M, K_groups, R)
                       .view(M, K_groups * R).view(torch.int32)
                       .reshape(M, K_groups))

    # tags: packed uint16 (4×4-bit). Use numpy uint16 to make NKI see uint16.
    tags_i32 = torch.zeros(M, K_groups, dtype=torch.int32)
    for k in range(R):
        tags_i32 |= (kept_indices[:, :, k].to(torch.int32) & 0xF) << (4 * k)
    tags_u16 = tags_i32.numpy().astype(np.uint16)                          # [M, K_g] uint16

    return stat_packed_i32, tags_u16, dense_sparse.to(torch.float8_e4m3fn)


def main() -> int:
    M = 64
    K_real = 512
    N = 128
    L, R = 16, 4
    K_packed = K_real // L

    print("=== Inputs ===")
    stat_packed_i32, tags_u16_np, dense_sparse_fp8 = build_sparse_fp8(M, K_real)
    _show("stat (packed i32 view of x4)", stat_packed_i32)
    print(f"  tags_u16 (numpy):             shape={tags_u16_np.shape}  dtype={tags_u16_np.dtype}  bytes={tags_u16_np.nbytes}")
    _show("dense_sparse_fp8 (reference)", dense_sparse_fp8)

    # moving: packed x4
    g2 = torch.Generator().manual_seed(7)
    mov_fp8 = (torch.randn(K_real, N, generator=g2) * 0.3).to(torch.float8_e4m3fn)
    K_packed_full = K_real // 4
    mov_packed_i32 = mov_fp8.view(torch.int32).reshape(K_packed_full, N).contiguous()
    _show("moving (packed i32 view of x4)", mov_packed_i32)

    # ============ NKI3 kernel ============
    print()
    print("=== Build NKI3 kernel ===")
    import nki
    import nki.language as nl
    import nki.isa as nisa
    from nki.dtype import float8_e4m3fn_x4 as FP8X4
    # private op from NKI2 private path
    from neuronxcc.nki._private.private_api import nc_matmul_sparse

    @nki.jit
    def sparse_kernel(stat_i32, tags_u16, mov_i32):
        # stat:  [P_stat=K_packed=32, F_stat=M=64]   (NKI2 matmul layout: P=contract)
        # mov:   [P_mov=K_packed=32, F_mov=N=128]
        # tags:  [P_tag=K_packed=32, F_tag=M=64]      uint16, 4×4-bit indices each
        P_stat, F_stat = stat_i32.shape
        P_mov, F_mov = mov_i32.shape
        P_tag, F_tag = tags_u16.shape

        out_ptr = nl.ndarray((F_stat, F_mov), dtype=nl.float32, buffer=nl.shared_hbm)
        stat = nl.load(stat_i32).view(FP8X4)
        tags = nl.load(tags_u16)
        mov = nl.load(mov_i32).view(FP8X4)

        psum = nl.zeros((F_stat, F_mov), dtype=nl.float32, buffer=nl.psum)
        psum[...] = nc_matmul_sparse(
            moving=mov, stationary=stat, tags=tags, compress_ratio=4
        )
        nl.store(out_ptr, value=psum)
        return out_ptr

    print()
    print("=== Run on Trainium (NKI3 baremetal) ===")
    # NKI3 baremetal
    try:
        from nki import baremetal as nki3_baremetal
        baremetal_fn = nki3_baremetal
    except ImportError:
        from neuronxcc.nki import baremetal as baremetal_fn

    # Layout: stat transposed for P=K_packed=32
    stat_T = stat_packed_i32.T.contiguous()                       # [32, 64]
    # tags: transpose so P=K_packed=32: tags_u16_np is [M=64, K_g=32] uint16
    tags_T = tags_u16_np.T.copy()                                  # [32, 64] uint16
    # moving: only first 32 partitions (matching contract)
    mov_for_sparse = mov_packed_i32[:K_packed, :].contiguous()    # [32, 128]
    _show("stat_T", stat_T)
    print(f"  tags_T:                       shape={tags_T.shape}  dtype={tags_T.dtype}  bytes={tags_T.nbytes}")
    _show("mov_for_sparse", mov_for_sparse)

    try:
        kn = baremetal_fn(sparse_kernel)
        t0 = time.perf_counter()
        out = kn(stat_T.numpy(), tags_T, mov_for_sparse.numpy())
        print(f"  ✅ ran in {time.perf_counter()-t0:.2f}s — out shape {out.shape} dtype {out.dtype}")
        print(f"  output stats: mean={out.mean():.4f} std={out.std():.4f} min={out.min():.4f} max={out.max():.4f}")
        # Reference: full dense_sparse @ slice of moving
        ref = (dense_sparse_fp8.float() @ mov_fp8[:K_real, :N].float())
        print(f"  reference shape: {ref.shape}")
        if out.shape == ref.shape:
            cos = torch.nn.functional.cosine_similarity(
                torch.from_numpy(out).flatten().unsqueeze(0),
                ref.flatten().unsqueeze(0)
            ).item()
            print(f"  cosine vs full sparse matmul ref: {cos:.4f}")
        return 0
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
