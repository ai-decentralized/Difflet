"""Micro-benchmark helpers for the CUDA backend.

Three pure functions so the DiT benchmark script (``scripts/cuda_bench_dit.py``)
stays thin and its bucketing can be unit-tested on a CPU host:

* ``categorize_kernel`` buckets a CUDA kernel name into gemm / attention / norm /
  elementwise / copy / other — the categories the profile table reports.
* ``summarize_profile`` turns ``(kernel_name, device_time_us)`` events (as from
  ``torch.profiler``'s ``key_averages()``) into totals, per-category totals and a top list.
* ``time_forward`` times a callable with CUDA events: warm-up runs untimed, then the
  median of ``iters`` runs — the "clean timed run" of the spec's three-runs rule.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Iterable
from typing import Any

import torch

CATEGORIES = ("gemm", "attention", "norm", "elementwise", "copy", "other")

# Order matters: the first matching bucket wins (copy before elementwise, so
# ``copy_kernel`` lands in "copy"; attention before gemm, so a cuDNN attention
# kernel that mentions "cutlass" still counts as attention).
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("copy", re.compile(r"Memcpy|Memset|copy_kernel|\bcopy\b", re.IGNORECASE)),
    ("attention", re.compile(r"flash|fmha|attention|sdpa|cudnn.*(fprop|attn)", re.IGNORECASE)),
    ("gemm", re.compile(r"gemm|cutlass|cublas|matmul|nvjet", re.IGNORECASE)),
    ("norm", re.compile(r"norm|layer_norm|rms|RowwiseMoments", re.IGNORECASE)),
    (
        "elementwise",
        re.compile(
            r"elementwise|vectorized|unrolled|reduce|softmax|index|where|cat\b|CatArray|fill",
            re.IGNORECASE,
        ),
    ),
)


_NOT_A_KERNEL = re.compile(r"^aten::|^cuda[A-Z]|Command Buffer Full|^ProfilerStep|^\[memory\]")


def is_kernel(name: str) -> bool:
    """True for a device kernel row of ``key_averages()``; False for the aten op rows
    (their device time is their kernels', counted once) and profiler markers."""
    return not _NOT_A_KERNEL.search(name)


def categorize_kernel(name: str) -> str:
    """Bucket a CUDA kernel name; see ``_PATTERNS`` for the order of precedence."""
    for category, pattern in _PATTERNS:
        if pattern.search(name):
            return category
    return "other"


def summarize_profile(events: Iterable[tuple[str, float]], *, top_n: int = 15) -> dict[str, Any]:
    """Totals in ms from ``(kernel_name, device_time_us)`` events.

    Returns ``{"total_ms", "by_category_ms": {category: ms} (non-zero only),
    "top": [{"name", "ms", "share"}]}`` with ``top`` sorted by time, descending.
    """
    by_name: dict[str, float] = {}
    for name, us in events:
        if not is_kernel(name):
            continue
        by_name[name] = by_name.get(name, 0.0) + float(us)
    total_us = sum(by_name.values())
    by_category: dict[str, float] = {}
    for name, us in by_name.items():
        cat = categorize_kernel(name)
        by_category[cat] = by_category.get(cat, 0.0) + us
    top = sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {
        "total_ms": total_us / 1000.0,
        "by_category_ms": {k: v / 1000.0 for k, v in by_category.items() if v > 0},
        "top": [
            {"name": n, "ms": us / 1000.0, "share": (us / total_us) if total_us else 0.0}
            for n, us in top
        ],
    }


def time_forward(fn: Callable[[], Any], *, warmup: int = 3, iters: int = 10) -> dict[str, Any]:
    """Median wall time of ``fn()`` on the GPU, in ms, after ``warmup`` untimed calls."""
    if not torch.cuda.is_available():
        raise RuntimeError("time_forward needs CUDA")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "iters": iters,
    }


__all__ = ["CATEGORIES", "categorize_kernel", "is_kernel", "summarize_profile", "time_forward"]
