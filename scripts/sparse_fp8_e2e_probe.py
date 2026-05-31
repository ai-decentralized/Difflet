#!/usr/bin/env python3
"""
GOAL: make nc_matmul_sparse actually run with fp8 packed data + tags and
produce a numerical result.

Approach (skip broken to_compressed_sparse helper, build everything by hand):
  1. Random fp8 weight  [M, K_real]
  2. Apply 16:4 mask: keep 4 of every 16 along K
  3. Per row, per group of 16: extract 4 nonzero fp8 values + their 4 4-bit indices
  4. Pack:
     - stationary: [M, K_real/16] of float8_e4m3fn_x4 (each elem = 4 fp8 packed)
     - tags:       [M, K_real/16] of uint16 (each elem packs 4 4-bit indices)
  5. Build moving: [P, F_mov] fp8 packed too (or whatever the ISA accepts)
  6. Call nc_matmul_sparse, examine result

We don't know the exact P-dim convention. Try the natural fit first:
  stationary [P_stat=M, F_stat=K_packed]   ← weight, M as partition
  moving     [P_mov=M, F_mov=N]            ← activation, same M as partition (??)
  tags       [P_tag=M, F_tag=K_packed]     ← same as stat
But verifier said all 3 partitions must match — so probably:
  stationary [P=K_packed_aligned, F=M_out]   ← weight transposed
  moving     [P=K_packed_aligned, F=N]
  tags       [P=K_packed_aligned, F=M_out]
We'll try and let the error guide us.
"""
from __future__ import annotations
import sys, time, traceback
import numpy as np
import torch


def _show(name, t):
    nb = t.element_size() * t.numel() if hasattr(t, 'element_size') else t.nbytes
    print(f"  {name:36s} shape={tuple(t.shape)}  dtype={t.dtype}  bytes={nb}")


def build_sparse_fp8(M: int, K_real: int, seed: int = 0):
    """Manually build sparse fp8 weight + matching tags + compressed packed view."""
    L, R = 16, 4
    assert K_real % L == 0
    K_groups = K_real // L                # number of 16-element groups along K
    K_packed = K_groups                   # K_c in float8_e4m3fn_x4 (each = 4 fp8 = 1 group's 4 nonzeros)

    g = torch.Generator().manual_seed(seed)
    # Use fp16 for math, quantize to fp8 at the end
    dense_fp16 = (torch.randn(M, K_real, generator=g, dtype=torch.float32) * 0.4)

    # Per (row, group): pick 4 of 16 nonzero indices, store the kept values + indices
    # tags: uint16, each element packs 4×4-bit indices (positions 0..15)
    # stationary: 4 fp8 values per (row, group), packed into 4 contiguous bytes -> int32
    g_idx = torch.Generator().manual_seed(seed + 1)
    kept_values_fp8 = torch.zeros(M, K_groups, R, dtype=torch.float32)
    kept_indices = torch.zeros(M, K_groups, R, dtype=torch.int64)
    for i in range(M):
        for gi in range(K_groups):
            # pick R=4 unique positions out of L=16
            perm = torch.randperm(L, generator=g_idx)
            chosen = perm[:R].sort().values
            kept_indices[i, gi] = chosen
            kept_values_fp8[i, gi] = dense_fp16[i, gi*L:(gi+1)*L][chosen]

    # Build the actual dense fp16 with zeros enforced (for reference matmul later)
    dense_sparse_fp16 = torch.zeros_like(dense_fp16)
    for i in range(M):
        for gi in range(K_groups):
            for k in range(R):
                pos = kept_indices[i, gi, k].item()
                dense_sparse_fp16[i, gi*L+pos] = kept_values_fp8[i, gi, k]

    # Quantize kept values to fp8
    kept_fp8 = kept_values_fp8.to(torch.float8_e4m3fn)            # [M, K_groups, R=4]

    # Pack 4 fp8 into one int32 (= float8_e4m3fn_x4 wire format)
    # Order: x4 layout packs along the "x4" dim contiguously
    stationary_packed_i32 = kept_fp8.view(torch.uint8).reshape(M, K_groups, R).view(M, K_groups * R)
    stationary_packed_i32 = stationary_packed_i32.view(torch.int32).reshape(M, K_groups)

    # Tags: try PACKED uint16 — 4 4-bit indices per element, matching weight x4 shape.
    # squeeze_tags docstring explicitly says ISA wants uint16 packed.
    tags_int32 = torch.zeros(M, K_groups, dtype=torch.int32)
    for k in range(R):
        tags_int32 |= (kept_indices[:, :, k].to(torch.int32) & 0xF) << (4 * k)
    # numpy view as uint16 (NKI sees uint16 dtype). torch only has int16 natively.
    tags_u16_np = tags_int32.numpy().astype('uint16')   # [M, K_groups] uint16
    tags_u16 = torch.from_numpy(tags_u16_np.view(np.int16))  # transport via int16 (same bits)

    # Re-quantize dense_sparse_fp16 to fp8 for reference
    dense_sparse_fp8 = dense_sparse_fp16.to(torch.float8_e4m3fn)

    return stationary_packed_i32, tags_u16, dense_sparse_fp8


def main() -> int:
    # Smallest valid shape
    M = 64
    K_real = 512    # must be divisible by R*128 = 512 per get_rand_mask, OK
    N = 128         # moving free dim
    L, R = 16, 4
    K_packed = K_real // L           # 32 packed x4 elements

    print(f"=== Build inputs ===")
    print(f"  M={M}  K_real={K_real}  N={N}  pattern={L}:{R}  K_packed={K_packed}")

    stat_packed_i32, tags_i16, dense_sparse_fp8 = build_sparse_fp8(M, K_real)
    _show("stationary packed (int32 view of x4)", stat_packed_i32)
    _show("tags (int16 packing 4 4-bit indices)", tags_i16)
    _show("dense_sparse_fp8 (reference)", dense_sparse_fp8)

    # moving (activation): also packed x4. shape [K_packed_full, N] in float8_x4
    # K_packed_full corresponds to the FULL K_real contracted per cycle.
    # If x4 packs 4 fp8 along K_real, then K_packed_full = K_real / 4 = 128
    K_packed_full = K_real // 4
    g2 = torch.Generator().manual_seed(7)
    mov_fp8 = (torch.randn(K_real, N, generator=g2) * 0.3).to(torch.float8_e4m3fn)
    # pack 4 fp8 contiguous along K into one int32 element
    mov_packed_i32 = mov_fp8.view(torch.int32).reshape(K_packed_full, N).contiguous()
    _show("moving packed (int32 view of x4)", mov_packed_i32)

    print()
    print("=== Build NKI kernel: try variant A — match all P dims ===")
    print("  layout: stat[P=M=64, F=K_packed=32], mov[P=M=64?, F=N], tags[P=M=64, F=K_packed=32]")
    # Note: verifier wants all P dims to match. Try P=M.
    # mov needs reshaping: original mov is [K_packed_full=128, N], use slice [M=64, N] for first try
    # This is wrong semantically but lets us see the verifier error first.

    from neuronxcc import nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki._private.private_api import nc_matmul_sparse
    from nki.dtype import float8_e4m3fn_x4 as FP8X4

    @nki.jit
    def sparse_kernel_v1(stat_i32, tags_i32, mov_i32):
        # nc_matmul_sparse returns local_tile (no dst=); use explicit slicing for AP
        P_stat, F_stat = stat_i32.shape
        P_mov, F_mov = mov_i32.shape
        P_tag, F_tag = tags_i32.shape
        out_ptr = nl.ndarray((F_stat, F_mov), dtype=nl.float32, buffer=nl.shared_hbm)

        stat = nl.load(stat_i32).view(FP8X4)
        tags = nl.load(tags_i32)
        mov = nl.load(mov_i32).view(FP8X4)

        # Pre-allocate psum, assign sliced result into it (gives AP hints)
        psum_buf = nl.zeros((F_stat, F_mov), dtype=nl.float32, buffer=nl.psum)
        psum_buf[0:F_stat, 0:F_mov] = nc_matmul_sparse(
            moving=mov[0:P_mov, 0:F_mov],
            stationary=stat[0:P_stat, 0:F_stat],
            tags=tags[0:P_tag, 0:F_tag],
            compress_ratio=4,
        )
        nl.store(out_ptr, value=psum_buf)
        return out_ptr

    print()
    print("=== Variant A: P=M for all 3 (semantically wrong but probes verifier) ===")
    from neuronxcc.nki import baremetal
    mov_slice = mov_packed_i32[:M, :N].contiguous()    # [64, 128]
    _show("mov_slice (used as variant A input)", mov_slice)
    try:
        kn = baremetal(sparse_kernel_v1)
        t0 = time.perf_counter()
        out = kn(stat_packed_i32.numpy(), tags_i16.numpy(), mov_slice.numpy())
        print(f"  ✅ ran in {time.perf_counter()-t0:.2f}s — out shape {out.shape} dtype {out.dtype}")
        print(f"  output stats: mean={out.mean():.3f} std={out.std():.3f}")
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
        # don't propagate; try variant B

    print()
    print("=== Variant B: stationary transposed so P=K_packed=32 (semantically correct) ===")
    print("  layout: stat[P=K_packed=32, F=M=64], mov[P=K_packed=32, F=N=128], tags[P=K_packed=32, F=M*R=256]")
    stat_T = stat_packed_i32.T.contiguous()                          # [32, 64]
    # tags: kept_indices was [M=64, K_groups=32, R=4]; want [K_groups=32, M=64, R=4] -> [32, 256]
    # We need to recompute since tags_u16 (= tags_unpacked_u8) is shape [M, K_groups*R] = [64, 128]
    # = [M, K_groups, R] flattened over (K_groups, R) on M side.
    # Convert back: reshape [M, K_groups, R] then permute and flatten
    tags_for_B = (tags_u16.reshape(M, K_packed, R)            # [M, K_g, R]
                  .permute(1, 0, 2)                            # [K_g, M, R]
                  .reshape(K_packed, M * R)                    # [K_g=32, M*R=256]
                  .contiguous())
    # moving: variant B uses K_packed_full=128 mov but with only 32 partitions -- this still
    # mismatches; per "contract dim" rule maybe mov also needs P=K_packed=32 from packed_full[:32]
    mov_for_B = mov_packed_i32[:K_packed, :].contiguous()   # [32, N=128]
    _show("stat_T", stat_T)
    _show("tags_for_B", tags_for_B)
    _show("mov_for_B", mov_for_B)
    try:
        kn = baremetal(sparse_kernel_v1)
        t0 = time.perf_counter()
        out = kn(stat_T.numpy(), tags_for_B.numpy(), mov_for_B.numpy())
        print(f"  ✅ ran in {time.perf_counter()-t0:.2f}s — out shape {out.shape} dtype {out.dtype}")
        print(f"  output stats: mean={out.mean():.3f} std={out.std():.3f}")
        # Reference: dense_sparse_fp8 [M=64, K_real=512] @ mov_fp8 [K_real=512, N=128]
        # but we only used 32 partitions of mov so this can't match — skip parity for B
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")

    print()
    print("=== Variant C: full mov partitions (K_packed_full=128), P=K_packed_full ===")
    print("  Hypothesis: maybe stat's P should expand to match mov, with tags encoding the compression")
    # Stationary: pad/replicate to P=K_packed_full?  Or compress moving?
    # Try: stationary shape [P=K_packed_full=128, F=M=64], tags [P=128, F=M=64]
    # Need to "decompress" stat/tags spatially... but we don't know how.
    # Skip for now.
    return 0


if __name__ == "__main__":
    sys.exit(main())
