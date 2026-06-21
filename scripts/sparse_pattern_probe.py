#!/usr/bin/env python3
"""
Empirical probe: does Trainium3 neuronx-cc auto-detect a hardware-compatible
16:4 sparsity pattern in a dense BF16 weight tensor and accelerate the matmul?

Hypothesis (from SDK static analysis):
- NISA MLIR dialect has no matmul_sparse op (only matmul / matmul_mx).
- neuronx-cc compile --help has no --enable-sparsity flag.
- Therefore dense bf16 weight with zeros should NOT auto-route to sparse ISA;
  forward time should be IDENTICAL to a random dense weight.

Method:
- One torch.matmul on XLA device, two weight variants:
    A) random_dense  : standard randn weight
    B) sparse_masked : same shape, hardware-compatible 16:4 mask applied
                       (12 of every 16 K-axis elements zeroed, layout per
                       neuronxcc.apis.sparsity.get_rand_mask)
- Both stored as dense bf16 (NO compression). Compiler sees identical HLO
  structure, only the weight constant values differ.
- Warmup, then time N forwards on each, compare medians.

Interpretation:
- |B - A| / A < 5%  -> no auto-sparse path (static analysis confirmed)
- B << A (~4x)      -> hidden auto-sparse codegen exists (jackpot)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_weight(M: int, K: int, seed: int, pattern: tuple[int, int] | None):
    """Build BF16 weight: random dense, or random with 16:4 hardware-compatible mask applied."""
    gen = torch.Generator().manual_seed(seed)
    # fp32 randn shifted away from 0 so bf16 rounding can't introduce stray zeros
    w = (torch.randn(M, K, generator=gen, dtype=torch.float32) + 1.0).to(torch.bfloat16)
    if pattern is None:
        return w, None
    # use the SDK's own mask generator -> guarantees hardware-compatible layout
    from neuronxcc.apis import sparsity

    mask = sparsity.get_rand_mask((M, K), pattern=pattern)
    w_masked = (w * mask.to(w.dtype)).contiguous()
    return w_masked, mask


def _run_xla(weight_cpu: torch.Tensor, x_cpu: torch.Tensor, *, warmup: int, repeats: int):
    """Move weight + x to XLA device, run torch.matmul N times, return per-iter times."""
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    weight = weight_cpu.to(device)
    x = x_cpu.to(device)
    xm.mark_step()  # force compile + initial load

    # warmup
    for _ in range(warmup):
        out = torch.matmul(x, weight)
        xm.mark_step()
        _ = out.cpu()  # force sync

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = torch.matmul(x, weight)
        xm.mark_step()
        _ = out.cpu()  # blocking sync forces measurement of true forward
        times.append(time.perf_counter() - t0)

    return times


def _stats(times: list[float]) -> dict[str, float]:
    return {
        "median_ms": 1000 * statistics.median(times),
        "min_ms": 1000 * min(times),
        "mean_ms": 1000 * statistics.mean(times),
        "stdev_ms": 1000 * statistics.stdev(times) if len(times) > 1 else 0.0,
        "n": len(times),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--batch", type=int, default=64,
                   help="leading dim on x (rows of matmul input)")
    p.add_argument("--pattern", type=str, default="16:4",
                   choices=("16:4", "4:8"))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metrics-out", type=Path,
                   default=Path("/tmp/difflet_sparse_pattern_probe_metrics.json"))
    args = p.parse_args()

    L, R = (int(x) for x in args.pattern.split(":"))
    pattern = (L, R)
    assert args.k % (R * 128) == 0, (
        f"K={args.k} must be divisible by ratio*{128} = {R*128} for {args.pattern} mask")

    # Sanity: confirm the dense-vs-masked test is well posed
    w_dense, _ = _build_weight(args.m, args.k, args.seed, pattern=None)
    w_sparse, mask = _build_weight(args.m, args.k, args.seed, pattern=pattern)
    actual_sparsity = 1.0 - (w_sparse != 0).float().mean().item()
    print(f"Weight shape: ({args.m}, {args.k}) bf16")
    print(f"Dense  nonzero fraction: {(w_dense != 0).float().mean().item():.4f}")
    print(f"Sparse nonzero fraction: {1.0 - actual_sparsity:.4f}  (target {R}/{L} = {R/L:.4f})")
    print(f"Storage: BOTH stored as dense bf16, only values differ.")
    print()

    # Same input batch for both
    gen = torch.Generator().manual_seed(args.seed + 1)
    x_cpu = (torch.randn(args.batch, args.m, generator=gen, dtype=torch.float32)
             .to(torch.bfloat16))

    print(f"=== Run A: dense random weight ===")
    times_A = _run_xla(w_dense, x_cpu, warmup=args.warmup, repeats=args.repeats)
    stats_A = _stats(times_A)
    print(f"  median {stats_A['median_ms']:.3f} ms  min {stats_A['min_ms']:.3f} ms  "
          f"stdev {stats_A['stdev_ms']:.3f} ms  n={stats_A['n']}")

    print()
    print(f"=== Run B: sparse-masked weight ({args.pattern} hardware-compatible) ===")
    times_B = _run_xla(w_sparse, x_cpu, warmup=args.warmup, repeats=args.repeats)
    stats_B = _stats(times_B)
    print(f"  median {stats_B['median_ms']:.3f} ms  min {stats_B['min_ms']:.3f} ms  "
          f"stdev {stats_B['stdev_ms']:.3f} ms  n={stats_B['n']}")

    print()
    print("=== Verdict ===")
    delta_pct = 100.0 * (stats_B["median_ms"] - stats_A["median_ms"]) / stats_A["median_ms"]
    ratio = stats_A["median_ms"] / stats_B["median_ms"]
    print(f"  sparse/dense median ratio: {stats_B['median_ms']/stats_A['median_ms']:.3f}x")
    print(f"  dense/sparse speedup:      {ratio:.3f}x")
    print(f"  delta:                     {delta_pct:+.1f}%")
    print()
    if abs(delta_pct) < 5:
        verdict = "NO_AUTO_SPARSE (within noise, identical execution)"
    elif ratio >= 1.5:
        verdict = "AUTO_SPARSE_DETECTED (sparse weight runs faster — investigate!)"
    elif ratio <= 0.8:
        verdict = "SPARSE_SLOWER (compiler may have triggered slow path on zeros)"
    else:
        verdict = "AMBIGUOUS (modest delta, re-run with more repeats)"
    print(f"  >>> {verdict} <<<")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps({
        "shape": {"M": args.m, "K": args.k, "N_unused": args.n, "batch": args.batch},
        "pattern": args.pattern,
        "actual_nonzero_fraction": 1.0 - actual_sparsity,
        "dense_random": stats_A,
        "sparse_masked": stats_B,
        "ratio_sparse_over_dense": stats_B["median_ms"] / stats_A["median_ms"],
        "speedup_dense_over_sparse": ratio,
        "delta_pct": delta_pct,
        "verdict": verdict,
    }, indent=2))
    print(f"\nMetrics: {args.metrics_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
