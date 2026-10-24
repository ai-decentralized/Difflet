"""Opt-in per-step wall-clock timing of a denoise loop.

Set ``DIFFLET_STEP_TIMES_FILE`` to a path and the denoise loops append one
JSON line per step: ``{"step", "step_s", "forward_s"}``. ``step_s`` is the
time since the previous step ended (the loop entry for step 0), ``forward_s``
the time inside the DiT call(s) of that step. Trainium DiT calls return host
tensors, so each forward is already synchronized. Unset, it is a no-op.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Iterator

STEP_TIMES_ENV = "DIFFLET_STEP_TIMES_FILE"


class StepTimer:
    def __init__(self, path: str) -> None:
        self.path = path
        self._last = time.perf_counter()
        self._forward = 0.0

    @classmethod
    def from_env(cls) -> "StepTimer | None":
        path = os.environ.get(STEP_TIMES_ENV)
        return cls(path) if path else None

    @contextmanager
    def forward(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._forward += time.perf_counter() - start

    def step_done(self, step_index: int) -> None:
        now = time.perf_counter()
        record = {"step": int(step_index), "step_s": now - self._last, "forward_s": self._forward}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self._last = now
        self._forward = 0.0


@contextmanager
def maybe_forward(timer: StepTimer | None) -> Iterator[None]:
    """``timer.forward()`` when timing is on, else nothing."""
    if timer is None:
        yield
    else:
        with timer.forward():
            yield
