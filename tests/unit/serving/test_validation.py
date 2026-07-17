from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

from difflet.serving.errors import DiffletServingError
from difflet.serving.validation import BoundedValidationExecutor


def test_preload_runs_before_executor_accepts_requests(monkeypatch):
    class _Validator:
        def __init__(self) -> None:
            self.preloaded = False
            self.thread_name = ""

        def preload(self) -> None:
            self.preloaded = True
            self.thread_name = threading.current_thread().name

    async def _run() -> None:
        validator = _Validator()
        executor = BoundedValidationExecutor(max_workers=1, max_waiting=0)
        await executor.start(validator)
        try:
            assert validator.preloaded is True
            assert validator.thread_name.startswith("difflet-validation")
            assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
            assert await executor.run(lambda: "ok") == "ok"
        finally:
            await executor.shutdown()

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    asyncio.run(_run())


def test_validation_capacity_is_physically_bounded():
    async def _run() -> None:
        executor = BoundedValidationExecutor(
            max_workers=2,
            max_waiting=1,
            timeout_s=2.0,
        )
        await executor.start()
        release = threading.Event()
        entered = 0
        entered_lock = threading.Lock()

        def block() -> None:
            nonlocal entered
            with entered_lock:
                entered += 1
            assert release.wait(timeout=3.0)

        tasks = [asyncio.create_task(executor.run(block)) for _ in range(3)]
        deadline = time.monotonic() + 1.0
        while executor.inflight != 3:
            if time.monotonic() >= deadline:
                raise AssertionError("three validation slots were not reserved")
            await asyncio.sleep(0)
        try:
            with pytest.raises(DiffletServingError) as full:
                await executor.run(block)
            assert (full.value.status_code, full.value.code) == (
                429,
                "validation_capacity_exhausted",
            )
        finally:
            release.set()
            await asyncio.gather(*tasks)
            await executor.shutdown()

    asyncio.run(_run())


def test_started_validation_keeps_slot_after_http_timeout():
    async def _run() -> None:
        executor = BoundedValidationExecutor(
            max_workers=1,
            max_waiting=0,
            timeout_s=0.02,
        )
        await executor.start()
        entered = threading.Event()
        release = threading.Event()

        def block() -> None:
            entered.set()
            assert release.wait(timeout=3.0)

        try:
            with pytest.raises(DiffletServingError) as timed_out:
                await executor.run(block)
            assert (timed_out.value.status_code, timed_out.value.code) == (
                504,
                "validation_timeout",
            )
            assert entered.is_set()
            assert executor.inflight == 1
            with pytest.raises(DiffletServingError) as full:
                await executor.run(lambda: None)
            assert full.value.code == "validation_capacity_exhausted"
            release.set()
            deadline = time.monotonic() + 1.0
            while executor.inflight:
                if time.monotonic() >= deadline:
                    raise AssertionError("completed validation did not release its slot")
                await asyncio.sleep(0)
        finally:
            release.set()
            await executor.shutdown()

    asyncio.run(_run())
