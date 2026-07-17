"""Bounded CPU request validation for serving endpoints."""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from difflet.serving.errors import DiffletServingError

_T = TypeVar("_T")


class BoundedValidationExecutor:
    """Run synchronous validators without blocking the ASGI event loop.

    ``ThreadPoolExecutor`` has an unbounded submission queue, so capacity is
    reserved before submission and remains owned until the underlying future
    actually finishes.  Timing out an HTTP waiter therefore cannot create more
    physical validation work than ``max_workers + max_waiting``.
    """

    def __init__(
        self,
        *,
        max_workers: int = 4,
        max_waiting: int = 32,
        timeout_s: float = 30.0,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
            raise ValueError("validation max_workers must be a positive integer")
        if isinstance(max_waiting, bool) or not isinstance(max_waiting, int) or max_waiting < 0:
            raise ValueError("validation max_waiting must be a nonnegative integer")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("validation timeout must be positive")
        self.max_workers = max_workers
        self.max_waiting = max_waiting
        self.timeout_s = float(timeout_s)
        self.capacity = max_workers + max_waiting
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="difflet-validation",
        )
        self._lock = threading.Lock()
        self._inflight = 0
        self._accepting = False
        self._closed = False

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    async def start(self, validator: Any | None = None) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("validation executor is closed")
            if self._accepting:
                return
        # HuggingFace tokenizers may otherwise create their own Rayon pool per
        # validation worker and defeat the explicit four-thread CPU budget.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        preload = getattr(validator, "preload", None)
        try:
            if callable(preload):
                await asyncio.wrap_future(self._executor.submit(preload))
        except BaseException:
            await self.shutdown()
            raise
        with self._lock:
            if not self._closed:
                self._accepting = True

    async def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._accepting = False
            self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    async def run(
        self,
        function: Callable[..., _T],
        /,
        *args: Any,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> _T:
        future = self._submit(function, *args, **kwargs)
        remaining = self.timeout_s
        if deadline is not None:
            remaining = min(remaining, deadline - time.monotonic())
        if remaining <= 0:
            future.cancel()
            raise _validation_timeout()
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), timeout=remaining)
        except asyncio.TimeoutError as exc:
            # ``cancel`` only removes work that has not started.  A running
            # future remains counted until its completion callback fires.
            future.cancel()
            raise _validation_timeout() from exc
        except asyncio.CancelledError:
            future.cancel()
            raise

    def _submit(self, function: Callable[..., _T], /, *args: Any, **kwargs: Any) -> Future[_T]:
        with self._lock:
            if not self._accepting or self._closed:
                raise DiffletServingError(
                    503,
                    "validation_unavailable",
                    "Request validation is not accepting work",
                    "server_error",
                )
            if self._inflight >= self.capacity:
                raise DiffletServingError(
                    429,
                    "validation_capacity_exhausted",
                    "Request validation capacity is exhausted",
                )
            self._inflight += 1
        try:
            future = self._executor.submit(function, *args, **kwargs)
        except BaseException:
            self._release_slot()
            raise
        future.add_done_callback(lambda _future: self._release_slot())
        return future

    def _release_slot(self) -> None:
        with self._lock:
            self._inflight = max(self._inflight - 1, 0)


def _validation_timeout() -> DiffletServingError:
    return DiffletServingError(
        504,
        "validation_timeout",
        "Request validation timed out",
        "server_error",
    )


__all__ = ["BoundedValidationExecutor"]
