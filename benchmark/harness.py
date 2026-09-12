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
class RealLoopStepTimer:
    """The cross-device per-step rule, in one place.

    Every device folder's ``DiT per-step`` column is defined the same way, and
    ``benchmark/step_realloop.py`` is where that definition is written down:

        inter-step deltas of a **real generate loop**, **device-synced**,
        **step 0 excluded**

    "Real generate" is load-bearing: it was adopted precisely to replace an
    isolated synthetic-input timer that used a different method per model, so
    its numbers were comparable neither to each other nor to the H100.

    Backends differ in exactly one thing -- what counts as a sync:

    ==========  =======================================================
    CUDA        ``torch.cuda.synchronize()``
    Trainium    nothing; the forward returns host tensors, so the
                post-call timestamp is already a sync point
    TPU         ``xm.wait_device_ops()``; without it the timestamp
                records when Python *enqueued* the step, not when the
                chip finished it
    ==========  =======================================================

    That last row is why this class exists. The rule was re-implemented in
    every adapter, and the TPU copy left the sync out -- reasonably, since
    syncing does cost a lazy backend more than an eager one, but the result was
    a published per-step figure smaller than the model's own attention. A
    backend that wants to argue for a different basis should report it
    *alongside* this one, not instead of it.
    """

    def __init__(self, sync: Optional[Callable[[], None]] = None):
        self._sync = sync
        self.stamps: list[float] = []

    def step(self) -> None:
        """Call once per denoise step, immediately after that step's DiT eval."""
        if self._sync is not None:
            self._sync()
        self.stamps.append(time.perf_counter())

    def deltas(self) -> list[float]:
        """Inter-step deltas. Step 0 is excluded: no delta covers it, and on a
        cold process it carries first-execution compilation."""
        return [b - a for a, b in zip(self.stamps, self.stamps[1:])]


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
    e2e_warm_breakdown: Optional[dict] = None  # the e2e_breakdown of the last warm run
    step_latency: Optional[dict] = None      # Stats as dict (per denoise step)
    step_basis: str = ""                     # how step_latency was measured
    step_latency_alt: dict[str, Any] = field(default_factory=dict)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    # TeaCache controller stats ({full_steps, skipped_steps, ...}) from the
    # same warm generate as stage_seconds; None when no mode was enabled.
    teacache: Optional[dict] = None
    # The "natural" pass: the same generate with no per-step instrumentation
    # sync. step_latency is for comparing devices; these are for knowing what a
    # real serving loop delivers. On an eager backend the two nearly coincide;
    # on a lazy one they do not, because the sync serialises tracing against
    # execution. Only populated by adapters advertising supports_natural_mode.
    e2e_warm_natural: Optional[dict] = None
    step_latency_natural: Optional[dict] = None
    # The request-only figure on a process that already holds the weights --
    # what a served request costs. ``e2e_warm`` is a fresh process on every
    # backend (trn2's CLI reloads weights each time; the TPU adapter restarts
    # its workers to match), so this is the other basis, reported alongside.
    # Its trn2 counterpart is e2e_warm_breakdown.compute_and_overhead_s.
    e2e_warm_resident: Optional[dict] = None
    # How the run was conducted: page cache dropped before the cold run,
    # discarded warm-up runs, iteration counts per basis.
    protocol: dict[str, Any] = field(default_factory=dict)
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
    #: Set True when ``run_generate`` accepts ``sync_steps=False`` and can run
    #: the denoise loop without a per-step device sync. The harness then
    #: reports that pass separately as the natural basis. Meaningful on every
    #: backend -- an eager one loses little to the sync, a lazy one loses a
    #: lot -- so implementing it everywhere is what would make the real-loop
    #: numbers comparable across devices too.
    supports_natural_mode: bool = False
    #: Set True when the adapter keeps the loaded model resident after
    #: ``run_generate`` and serves further requests on it via ``run_request``.
    #: ``run_generate`` itself is "fresh process -> one output" on every
    #: backend; that is what keeps e2e cold/warm comparable across devices.
    supports_resident_mode: bool = False

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
        """Run one full end-to-end generate in a FRESH process (weights loaded
        again). Return a dict with at least ``wall_seconds`` and optionally
        ``load_seconds``, ``e2e_breakdown``, ``step_seconds`` (list),
        ``peak_mem_gb``, ``output`` (OutputInfo dict), and ``log`` (path)."""
        raise NotImplementedError

    def run_request(self, spec, sync_steps: bool = True) -> dict:
        """One more request on the process ``run_generate`` left resident
        (``supports_resident_mode`` only). Same result shape as ``run_generate``
        minus the load."""
        raise NotImplementedError

    def tag(self, name: str) -> None:
        """Label the next run; adapters that save per-run media use it."""

    def shutdown(self) -> None:
        """Release resident processes / devices. No-op by default."""


def drop_page_cache() -> bool:
    """``sync; echo 3 > /proc/sys/vm/drop_caches`` through passwordless sudo.

    The trn2 protocol (``benchmark/cold_warm_e2e.py``): the cold run's weight
    load has to be a real disk read, otherwise "cold" means whatever the page
    cache happened to hold. Returns False when sudo is unavailable, and the
    caller records that in the result instead of pretending.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, timeout=900,
        )
    except Exception:  # noqa: BLE001 - no sudo, no sh, timeout: all "not dropped"
        return False
    return proc.returncode == 0
