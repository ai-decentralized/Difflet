#!/usr/bin/env python3
"""Is splash attention worth wiring in? Sweep its block sizes at real shapes.

Out of the box at 4096 tokens / 24 heads it measured 1.5x SLOWER than SDPA,
with every block_* left at its default of 512. This sweeps them against the
shapes difflet actually runs.
"""

import itertools
import time

import torch
import torch_tpu

dev = torch.device("tpu")
SHAPES = [
    ("qwen 1024^2", 4096, 24, 128),
    ("qwen + text", 4608, 24, 128),
    ("flux 1024^2", 4608, 24, 128),
]
BLOCKS = [256, 512, 1024, 2048]


def bench(fn, iters=8):
    fn(); torch.accelerator.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - t) / iters * 1000


for label, S, H, D in SHAPES:
    torch.manual_seed(0)
    q = torch.randn(1, H, S, D, dtype=torch.bfloat16).to(dev)
    k = torch.randn(1, H, S, D, dtype=torch.bfloat16).to(dev)
    v = torch.randn(1, H, S, D, dtype=torch.bfloat16).to(dev)
    torch.accelerator.synchronize()

    base = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    print(f"\n### {label}  S={S} H={H} D={D}   SDPA = {base:.2f} ms")
    best = (None, 1e9)
    for bq, bkv in itertools.product(BLOCKS, BLOCKS):
        try:
            fn = lambda: torch.tpu.ops.splash_sdpa(
                q, k, v, is_causal=False, block_q=bq, block_kv=bkv, block_kv_compute=bkv
            )
            ms = bench(fn)
        except Exception as exc:
            print(f"  block_q={bq:<5} block_kv={bkv:<5} FAILED {type(exc).__name__}")
            continue
        flag = "  <-- best" if ms < best[1] else ""
        if ms < best[1]:
            best = ((bq, bkv), ms)
        print(f"  block_q={bq:<5} block_kv={bkv:<5} {ms:7.2f} ms  {base/ms:5.2f}x vs SDPA{flag}")
    if best[0]:
        print(f"  BEST {best[0]} -> {best[1]:.2f} ms ({base/best[1]:.2f}x vs SDPA)")
