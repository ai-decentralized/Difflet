"""Backend-generic benchmark harness for diffusion models.

The harness is intentionally backend-agnostic: it defines a small ``BackendAdapter``
contract and a set of universal metrics, and knows nothing about Trainium, CUDA, or
CPU specifically. Each backend ships an adapter (see ``benchmark/adapters/``) that
implements download → compile → load → infer; the harness times those phases the
same way for every backend so results are directly comparable.

Universal metrics (all backends report the same shape):
  * ``compile_seconds``      — ahead-of-time build/trace/compile (0 for eager backends)
  * ``load_seconds``         — weights load onto the device
  * ``e2e_cold_seconds``     — first full generate (incl. one-time costs in scope)
  * ``e2e_warm_seconds``     — steady-state full generate (Stats over N iters)
  * ``step_seconds``         — per denoising-step latency (Stats), the core compute
  * ``throughput``           — backend-defined unit/s (e.g. steps/s, frames/s, img/s)
  * ``peak_device_mem_gb``   — peak accelerator memory during inference
  * ``output``               — shape / dtype / finite / value-range of the result

Nothing here imports torch or any backend SDK, so it is import-safe everywhere.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    """Summary statistics over repeated timed measurements (seconds)."""

    n: int
    mean: float
    median: float
    std: float
    p50: float
    p90: float
    min: float
    max: float
    samples: list[float] = field(default_factory=list)

    @classmethod
    def from_samples(cls, samples: list[float]) -> "Stats":
        s = sorted(samples)
        n = len(s)
        if n == 0:
            return cls(0, 0, 0, 0, 0, 0, 0, 0, [])

        def pct(p: float) -> float:
            if n == 1:
                return s[0]
            idx = min(n - 1, max(0, int(round(p * (n - 1)))))
            return s[idx]

        return cls(
            n=n,
            mean=statistics.fmean(s),
            median=statistics.median(s),
            std=statistics.pstdev(s) if n > 1 else 0.0,
            p50=pct(0.50),
            p90=pct(0.90),
            min=s[0],
            max=s[-1],
            samples=samples,
        )


def timed(fn: Callable[[], Any], *, warmup: int = 1, iters: int = 5,
          sync: Optional[Callable[[], None]] = None) -> tuple[Stats, Any]:
    """Run ``fn`` ``warmup``+``iters`` times, return (Stats over iters, last result).

    ``sync`` (optional) is called after each invocation to force the backend to
    finish async work before the timer stops (e.g. device synchronize).
    """
    last = None
    for _ in range(max(0, warmup)):
        last = fn()
        if sync:
            sync()
    samples: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        last = fn()
        if sync:
            sync()
        samples.append(time.perf_counter() - t0)
    return Stats.from_samples(samples), last


# --------------------------------------------------------------------------- #
# Metrics container
# --------------------------------------------------------------------------- #
@dataclass
class OutputInfo:
    shape: Optional[list[int]] = None
    dtype: Optional[str] = None
    finite: Optional[bool] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    note: str = ""


@dataclass
class BenchResult:
    # identity
    model_id: str
    model_type: str
    backend: str
    device: str = ""
    dtype: str = ""
    parallel: dict[str, int] = field(default_factory=dict)
    shape: dict[str, Any] = field(default_factory=dict)
    steps: Optional[int] = None
    # phase timings (seconds)
    compile_seconds: Optional[float] = None
    compile_breakdown: dict[str, float] = field(default_factory=dict)
    load_seconds: Optional[float] = None     # total weights load across all stages
    e2e_cold_seconds: Optional[float] = None
    e2e_breakdown: Optional[dict] = None     # per-stage load/compute split of cold e2e
    e2e_warm: Optional[dict] = None          # Stats as dict
    step_latency: Optional[dict] = None      # Stats as dict (per denoise step)
    throughput: dict[str, float] = field(default_factory=dict)
    peak_device_mem_gb: Optional[float] = None
    output: Optional[dict] = None            # OutputInfo as dict
    # provenance
    toolchain: dict[str, str] = field(default_factory=dict)
    config_label: str = ""                   # the "best perf version" knobs used
    status: str = "ok"                       # ok | partial | failed | pending
    notes: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Backend adapter contract
# --------------------------------------------------------------------------- #
class BackendAdapter:
    """Contract every backend implements. The harness only ever calls these.

    A backend is "compiled" (AOT: Trainium, TensorRT) or "eager" (CUDA/CPU via
    a framework). Eager backends return ``compile_seconds=0`` and make ``compile``
    a no-op. ``measure`` orchestrates the universal phases; subclasses normally only
    implement the hooks below.
    """

    name: str = "base"

    def device_info(self) -> str:  # human-readable accelerator description
        raise NotImplementedError

    def toolchain(self) -> dict[str, str]:  # versions for provenance
        return {}

    def prepare(self, spec) -> None:  # download weights, etc.
        raise NotImplementedError

    def compile(self, spec) -> tuple[float, dict[str, float]]:
        """Return (total_compile_seconds, breakdown_by_component)."""
        raise NotImplementedError

    def run_generate(self, spec) -> dict:
        """Run one full end-to-end generate. Return a dict with at least
        ``wall_seconds`` and optionally ``load_seconds``, ``step_seconds`` (list),
        ``peak_mem_gb``, ``output`` (OutputInfo dict), and ``log`` (path)."""
        raise NotImplementedError
