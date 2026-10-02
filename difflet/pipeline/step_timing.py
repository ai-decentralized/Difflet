"""Per-step DiT wall-time reporting shared by every model's denoise loop.

The benchmark adapter (``benchmark/adapters/trainium.py``) reads two lines from
the generate log: the raw per-step list (``[<tag>] dit-step-seconds: [...]``,
step 0 dropped by the adapter — the cross-device per-step rule in
``benchmark/harness.py``) and a human summary. The Neuron forward returns host
tensors, so a ``perf_counter`` span around it is device-synced; TeaCache-skipped
steps are not timed (they run no DiT).
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Iterator


def format_dit_step_report(tag: str, step_seconds: list[float]) -> str:
    raw = f"[{tag}] dit-step-seconds: [" + ", ".join(f"{s:.4f}" for s in step_seconds) + "]"
    tail = step_seconds[1:]
    if not tail:
        return raw + f"\n[{tag}] dit-step ms: n=0 (fewer than two DiT steps; no per-step stat)"
    ms = [s * 1000.0 for s in tail]
    summary = (
        f"[{tag}] dit-step ms: n={len(ms)} mean={statistics.fmean(ms):.1f} "
        f"median={statistics.median(ms):.1f} min={min(ms):.1f} max={max(ms):.1f} "
        "(step 0 excluded)"
    )
    return raw + "\n" + summary


class DiTStepTimer:
    """Collects one wall-time sample per timed step; ``report()`` renders the two lines."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.seconds: list[float] = []

    @contextmanager
    def step(self) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds.append(time.perf_counter() - started)

    def report(self) -> str:
        return format_dit_step_report(self.tag, self.seconds)


__all__ = ["DiTStepTimer", "format_dit_step_report"]
