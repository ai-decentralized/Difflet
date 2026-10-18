"""`difflet serve --dp-degree N` on CUDA: N resident engines, one GPU each, behind one engine
(CUDA plan F2b). The dispatch and lifecycle are proven with the fake worker on CPU; the GPU
pinning lives in the resident worker's environment (test_resident_worker_engine.py)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest

from difflet.serving.engines.replicated import ReplicatedServingEngine
from difflet.serving.engines.resident_worker import (
    ResidentWorkerConfig,
    ResidentWorkerServingEngine,
)
from tests.unit.serving.test_resident_worker_engine import _request, _runtime

FAKE = "tests.unit.serving.fake_worker:FakeServingOrchestrator"


def _replica(index: int) -> ResidentWorkerServingEngine:
    runtime = _runtime()
    plan = runtime.runtime_plan
    runtime = replace(
        runtime,
        runtime_plan=replace(
            plan, environment=replace(plan.environment, available_core_ids=(index,))
        ),
    )
    return ResidentWorkerServingEngine(
        runtime=runtime,
        orchestrator_factory=FAKE,
        config=ResidentWorkerConfig(request_timeout=5, worker_restart_timeout=5),
    )


def test_two_concurrent_requests_run_on_two_replicas():
    async def _run():
        engine = ReplicatedServingEngine([_replica(0), _replica(1)])
        assert not engine.ready
        await engine.start()
        try:
            assert engine.ready and engine.healthy
            started = time.monotonic()
            outputs = await asyncio.gather(
                engine.generate(_request("req-1", "sleep:0.4")),
                engine.generate(_request("req-2", "sleep:0.4")),
            )
            wall = time.monotonic() - started
        finally:
            await engine.shutdown()
        return outputs, wall, engine

    outputs, wall, engine = asyncio.run(_run())
    assert [o.data for o in outputs] == [b"sleep:0.4:64x64", b"sleep:0.4:64x64"]
    # Serial on one worker would be >= 0.8 s; two workers run them side by side.
    assert wall < 0.7, wall
    assert not engine.ready and not engine.healthy


def test_a_dead_replica_makes_the_engine_not_ready():
    async def _run():
        engine = ReplicatedServingEngine([_replica(0), _replica(1)])
        await engine.start()
        try:
            engine.replicas[1]._worker.terminate()
            return engine.ready, engine.healthy
        finally:
            await engine.shutdown()

    ready, healthy = asyncio.run(_run())
    assert ready is False and healthy is False


class _StubReplica:
    def __init__(self, name: str, pending: int = 0) -> None:
        self.name = name
        self.pending_requests = pending
        self.calls: list[str] = []
        self.started = False
        self.stopped = False
        self.ready = True
        self.healthy = True

    async def start(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.stopped = True

    async def generate(self, request):
        self.calls.append(request.request_id)
        return request.request_id


def test_dispatch_goes_to_the_replica_with_the_fewest_pending_then_round_robin():
    a, b, c = _StubReplica("a", pending=2), _StubReplica("b"), _StubReplica("c")
    engine = ReplicatedServingEngine([a, b, c])

    async def _run():
        out = []
        for i in range(4):
            out.append(await engine.generate(_request(f"req-{i}", "x")))
        return out

    assert asyncio.run(_run()) == ["req-0", "req-1", "req-2", "req-3"]
    # b and c tie at 0 pending (the stubs never count): alternate between them; a (2) is never
    # chosen while a less loaded replica exists.
    assert a.calls == []
    assert b.calls == ["req-0", "req-2"] and c.calls == ["req-1", "req-3"]


def test_dispatch_skips_a_replica_that_is_not_ready():
    # A recovering replica answers 503 at once, so its pending count never grows; it must not
    # be ranked while a ready one exists.
    a, b = _StubReplica("a"), _StubReplica("b")
    b.ready = False
    engine = ReplicatedServingEngine([a, b])

    async def _run():
        for i in range(3):
            await engine.generate(_request(f"req-{i}", "x"))

    asyncio.run(_run())
    assert a.calls == ["req-0", "req-1", "req-2"] and b.calls == []
    # No ready replica at all: the request still reaches one (and gets its own 503 there).
    a.ready = False
    asyncio.run(engine.generate(_request("req-3", "x")))
    assert "req-3" in a.calls + b.calls


def test_a_replica_that_fails_to_start_shuts_the_others_down():
    class _Failing(_StubReplica):
        async def start(self) -> None:
            raise RuntimeError("no GPU 1")

    a, b = _StubReplica("a"), _Failing("b")
    engine = ReplicatedServingEngine([a, b])
    with pytest.raises(RuntimeError, match="no GPU 1"):
        asyncio.run(engine.start())
    assert a.started and a.stopped and b.stopped


def test_lifecycle_fans_out_and_aggregates_state():
    a, b = _StubReplica("a"), _StubReplica("b")
    engine = ReplicatedServingEngine([a, b])
    asyncio.run(engine.start())
    assert a.started and b.started
    assert engine.ready and engine.healthy
    b.ready = False
    assert not engine.ready and engine.healthy
    b.healthy = False
    assert not engine.healthy
    asyncio.run(engine.shutdown())
    assert a.stopped and b.stopped
    assert len(engine.replicas) == 2


def test_at_least_one_replica_is_required():
    with pytest.raises(ValueError):
        ReplicatedServingEngine([])
