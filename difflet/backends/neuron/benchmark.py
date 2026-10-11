"""Benchmark harness: comparable latency numbers across neuron eager, neuron compiled and AoT.

Timing rules, each removing one source of error:

* synchronise before and after every timed call: device execution is
  asynchronous, so an unsynchronised clock measures dispatch, not execution;
* the first call is timed separately and excluded: it includes compilation
  (per op in eager, per graph when compiled, artifact load on AoT);
* cases run in interleaved rounds (A B A B ...), so slow drift such as thermal
  throttling affects every case alike instead of the last case alone;
* two stability flags: ``spread`` (p90 - p10) / median over all samples, and
  ``drift`` max / min of the per-round medians. A drift flag means the device
  speed changed during the run, so the numbers are not comparable;
* the allocator cache is released between cases, so one case's freed blocks do
  not change the next case's allocations.

The synchronisation function is a parameter (default: ``torch.neuron.synchronize``)
so the same harness runs on the AoT stack with an XLA synchronisation.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from difflet.backends.neuron.buildinfo import build_info

SPREAD_LIMIT = 0.10
DRIFT_LIMIT = 1.10


@dataclass
class CaseResult:
    name: str
    first_call_ms: float
    median_ms: float
    p10_ms: float
    p90_ms: float
    spread: float
    drift: float
    round_medians_ms: list[float]
    samples_ms: list[float] = field(repr=False)
    peak_memory: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        line = (f"{self.name:32s} median {self.median_ms:9.2f} ms  p10 {self.p10_ms:9.2f}  p90 {self.p90_ms:9.2f}  "
                f"first {self.first_call_ms:10.1f} ms  spread {self.spread:5.1%}  drift {self.drift:.3f}")
        return line + (f"  [{', '.join(self.flags)}]" if self.flags else "")


def neuron_sync() -> None:
    import torch

    torch.neuron.synchronize()


def run(
    cases: Mapping[str, Callable[[], Any]],
    *,
    warmup: int = 3,
    iters: int = 10,
    rounds: int = 5,
    sync: Callable[[], None] | None = None,
    release_cache: Callable[[], None] | None = None,
    memory_stats: Callable[[], Mapping[str, Any]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[CaseResult]:
    """Time each zero-argument callable; ``iters`` timed calls per case per round."""
    sync = sync or neuron_sync
    names = list(cases)
    first: dict[str, float] = {}
    for name in names:  # first call (compile or load), then warm-up, per case
        _release(release_cache)
        first[name] = _timed(cases[name], sync, clock)
        for _ in range(warmup):
            cases[name]()
        sync()

    per_round: dict[str, list[list[float]]] = {name: [] for name in names}
    peaks: dict[str, dict[str, int]] = {name: {} for name in names}
    for _ in range(rounds):
        for name in names:
            _release(release_cache)
            per_round[name].append([_timed(cases[name], sync, clock) for _ in range(iters)])
            peaks[name] = _peaks(memory_stats, peaks[name])

    return [_summarise(name, first[name], per_round[name], peaks[name]) for name in names]


def report(results: list[CaseResult], config: Mapping[str, Any] | None = None) -> str:
    """Print the table; return JSON with the build, the configuration and every sample."""
    for result in results:
        print(result, flush=True)
    record = {"build": build_info(), "config": dict(config or {}), "results": [asdict(r) for r in results]}
    print(f"build: {record['build']}", flush=True)
    return json.dumps(record, indent=2)


def neuron_release_cache() -> None:
    import torch_neuronx

    torch_neuronx.empty_cache()


def neuron_memory_stats() -> Mapping[str, Any]:
    import torch_neuronx

    return torch_neuronx.memory_stats()


# ---- internals -----------------------------------------------------------------------


def _timed(fn, sync, clock) -> float:
    sync()
    start = clock()
    fn()
    sync()
    return (clock() - start) * 1e3


def _release(release_cache) -> None:
    if release_cache is not None:
        release_cache()


def _peaks(memory_stats, current: dict[str, int]) -> dict[str, int]:
    if memory_stats is None:
        return current
    stats = memory_stats() or {}
    for key, value in stats.items():
        if "peak" in str(key) and isinstance(value, (int, float)):
            current[str(key)] = max(int(value), current.get(str(key), 0))
    return current


def _quantile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _summarise(name, first_ms, rounds_ms, peak_memory) -> CaseResult:
    samples = [t for r in rounds_ms for t in r]
    ordered = sorted(samples)
    median = statistics.median(ordered)
    p10, p90 = _quantile(ordered, 0.10), _quantile(ordered, 0.90)
    round_medians = [statistics.median(r) for r in rounds_ms]
    spread = (p90 - p10) / median if median > 0 else 0.0
    drift = max(round_medians) / min(round_medians) if min(round_medians) > 0 else 1.0
    flags = []
    if spread > SPREAD_LIMIT:
        flags.append("unstable")
    if drift > DRIFT_LIMIT:
        flags.append("drift: device speed changed during the run")
    return CaseResult(name=name, first_call_ms=first_ms, median_ms=median, p10_ms=p10, p90_ms=p90,
                      spread=spread, drift=drift, round_medians_ms=round_medians, samples_ms=samples,
                      peak_memory=peak_memory, flags=flags)


__all__ = [
    "CaseResult",
    "DRIFT_LIMIT",
    "SPREAD_LIMIT",
    "neuron_memory_stats",
    "neuron_release_cache",
    "neuron_sync",
    "report",
    "run",
]
