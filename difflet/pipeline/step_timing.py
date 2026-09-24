"""Opt-in per-step wall-clock timing for the denoise loop.

The step-caching table needs a denoise-loop figure that *includes* the steps the
cache skipped, so it cannot be derived from a DiT-forward benchmark (which times
executed calls only) nor from two end-to-end runs subtracted (end-to-end is
weight-load dominated: benchmark/step_latency.py documents an LTX-2 pair that
measured 293 s and 628 s for the same configuration).

Enabled only when ``DIFFLET_STEP_TIMING`` is set, so no measured path changes
unless a measurement run asks for it. Timing a step means reading the clock
around work the caller already does; nothing is synchronised or transferred that
the loop did not already require.

Set ``DIFFLET_STEP_TIMING_OUT`` to also write the per-step series as JSON.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


def enabled() -> bool:
    return bool(os.environ.get("DIFFLET_STEP_TIMING"))


class StepTimer:
    """Collect per-step wall times and print one summary line at the end."""

    def __init__(self, label: str, total_steps: int) -> None:
        self.label = label
        self.total_steps = int(total_steps)
        self.step_ms: list[float] = []
        self._start: float | None = None

    def __enter__(self) -> "StepTimer":
        self._start = time.monotonic()
        return self

    def tick(self) -> None:
        """Record the boundary between one step and the next."""
        now = time.monotonic()
        if self._start is None:
            self._start = now
            return
        previous = self._start + sum(self.step_ms) / 1000.0
        self.step_ms.append((now - previous) * 1000.0)

    def __exit__(self, *exc: Any) -> None:
        if self._start is None:
            return
        total_s = time.monotonic() - self._start
        steps = self.total_steps or len(self.step_ms)
        per_step = (total_s * 1000.0 / steps) if steps else 0.0
        # Step 0 carries first-call effects, so report the median of the rest
        # alongside the mean the loop column needs.
        tail = sorted(self.step_ms[1:]) or sorted(self.step_ms)
        median = tail[len(tail) // 2] if tail else 0.0
        print(
            f"[steptiming] {self.label}: loop={total_s:.3f}s steps={steps} "
            f"mean={per_step:.1f}ms/step median_excl_step0={median:.1f}ms",
            flush=True,
        )
        out = os.environ.get("DIFFLET_STEP_TIMING_OUT")
        if out:
            path = Path(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "label": self.label,
                        "loop_s": round(total_s, 6),
                        "steps": steps,
                        "mean_ms_per_step": round(per_step, 3),
                        "median_ms_excl_step0": round(median, 3),
                        "step_ms": [round(v, 3) for v in self.step_ms],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[steptiming] wrote {path}", flush=True)


def timed_steps(label: str, timesteps: Iterable[Any]) -> Iterator[tuple[int, Any]]:
    """Wrap ``enumerate(timesteps)``, timing each step when enabled.

    Yields the same ``(index, timestep)`` pairs either way, so the caller reads
    identically whether or not timing is on.
    """
    steps = list(timesteps)
    if not enabled():
        yield from enumerate(steps)
        return
    with StepTimer(label, len(steps)) as timer:
        for index, timestep in enumerate(steps):
            yield index, timestep
            timer.tick()
