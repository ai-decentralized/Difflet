from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from pathlib import Path

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.errors import DiffletServingError
from difflet.serving.engines.resident_worker import (
    ResidentWorkerConfig,
    ResidentWorkerServingEngine,
    _apply_worker_runtime_environment,
    _error_from_reply,
    _reply_from_error,
)
from difflet.serving.types import (
    ArtifactBinding,
    ArtifactSet,
    CompileArtifactIdentity,
    DiffletCompileSpec,
    DiffletGenerateRequest,
    DistributedProcessEnvironment,
    ParallelTopology,
    PipelineDefinition,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    RuntimeEnvironment,
    RuntimePlan,
    ServingProfile,
    StageDefinition,
    StageRuntimeSpec,
    WorkerAllocationSpec,
)

_WORKER_ENVIRONMENT_NAMES = (
    "NEURON_RT_VISIBLE_CORES",
    "NEURON_RT_NUM_CORES",
    "NEURON_RT_VIRTUAL_CORE_SIZE",
    "NEURON_LOGICAL_NC_CONFIG",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
)


@pytest.fixture
def preserve_worker_environment():
    original = {name: os.environ.get(name) for name in _WORKER_ENVIRONMENT_NAMES}
    yield
    for name, value in original.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _profile() -> ServingProfile:
    return ServingProfile(
        model_id="fake/model",
        model_type="fake",
        height=64,
        width=64,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=1),
    )


def _runtime() -> ResolvedRuntimeBundle:
    profile = _profile()
    pipeline = PipelineDefinition(
        model_type="fake",
        stages=(
            StageDefinition(
                stage_id="pipeline",
                kind="opaque_pipeline",
                role="pipeline",
                final_output=True,
            ),
        ),
    )
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment((0,), None, None, None, distributed, distributed)
    allocation = WorkerAllocationSpec("fake-worker", 1, 1, 1)
    topology = ParallelTopology(1, 1, 1)
    identity = CompileArtifactIdentity.from_cache_inputs({"model": "fake"})
    spec = DiffletCompileSpec("pipeline", "pipeline", identity)
    path = Path("/tmp/difflet-fake-artifact")
    binding = ArtifactBinding(
        "pipeline",
        path,
        path / "difflet_generation_manifest.json",
        identity,
        "g0000000000000001",
        "0" * 64,
    )
    return ResolvedRuntimeBundle(
        profile=profile,
        source=ResolvedModelSource("hf_snapshot", "fake/model", None, "/tmp/fake-model", "a" * 40),
        pipeline_definition=pipeline,
        runtime_plan=RuntimePlan(
            "resident",
            identity.digest,
            environment,
            (allocation,),
            (StageRuntimeSpec("pipeline", allocation.allocation_id, topology, "pipeline"),),
        ),
        compile_specs=(spec,),
        artifacts=ArtifactSet((binding,)),
    )


def _request(request_id: str, prompt: str) -> DiffletGenerateRequest:
    return DiffletGenerateRequest(
        request_id=request_id,
        model="fake/model",
        prompt=prompt,
        height=64,
        width=64,
        num_inference_steps=1,
        guidance_scale=1.0,
        seed=0,
    )


def test_resident_worker_engine_generates_with_child_process():
    async def _run():
        engine = ResidentWorkerServingEngine(
            runtime=_runtime(),
            orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
            config=ResidentWorkerConfig(request_timeout=5, worker_restart_timeout=5),
        )
        await engine.start()
        try:
            out = await engine.generate(_request("req-1", "hello"))
        finally:
            await engine.shutdown()
        return out

    output = asyncio.run(_run())

    assert output.data == b"hello:64x64"
    assert output.mime_type == "image/png"


def test_resident_worker_startup_failure_closes_per_spawn_resources():
    async def _run():
        engine = ResidentWorkerServingEngine(
            runtime=_runtime(),
            orchestrator_factory="tests.unit.serving.fake_worker:MissingOrchestrator",
            config=ResidentWorkerConfig(worker_restart_timeout=5),
        )
        with pytest.raises(DiffletServingError) as exc:
            await engine.start()
        return engine, exc.value

    engine, error = asyncio.run(_run())

    assert error.code == "internal_error"
    assert engine._worker._process is None
    assert engine._worker._status_thread is None
    assert engine._worker._cmd_q is None
    assert engine._worker._cancel_q is None
    assert engine._worker._reply_q is None


def test_resident_worker_applies_runtime_environment_before_orchestrator(monkeypatch):
    async def _run():
        runtime = _runtime()
        allocation = replace(
            runtime.runtime_plan.allocations[0],
            requested_virtual_core_size=2,
            effective_virtual_core_size=2,
            requested_logical_nc_config=1,
            effective_logical_nc_config=1,
        )
        runtime = replace(
            runtime,
            runtime_plan=replace(runtime.runtime_plan, allocations=(allocation,)),
        )
        engine = ResidentWorkerServingEngine(
            runtime=runtime,
            orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
            config=ResidentWorkerConfig(request_timeout=5, worker_restart_timeout=5),
        )
        await engine.start()
        try:
            return await engine.generate(_request("runtime-env", "runtime-env"))
        finally:
            await engine.shutdown()

    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "9")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "9")
    monkeypatch.setenv("NEURON_RT_VIRTUAL_CORE_SIZE", "9")
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "9")
    monkeypatch.setenv("WORLD_SIZE", "9")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "9")
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("LOCAL_RANK", "8")

    output = asyncio.run(_run())

    assert output.data == b"0:1:2:1:1:1:0:0"


def test_worker_environment_clears_unspecified_optional_neuron_settings(
    monkeypatch, preserve_worker_environment
):
    monkeypatch.setenv("NEURON_RT_VIRTUAL_CORE_SIZE", "9")
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "9")

    _apply_worker_runtime_environment(_runtime())

    assert "NEURON_RT_VIRTUAL_CORE_SIZE" not in os.environ
    assert "NEURON_LOGICAL_NC_CONFIG" not in os.environ


def test_worker_environment_applies_partitioned_core_assignment(preserve_worker_environment):
    runtime = _runtime()
    environment = replace(runtime.runtime_plan.environment, available_core_ids=(4, 5, 6, 7))
    allocation = replace(
        runtime.runtime_plan.allocations[0],
        requested_num_cores=4,
        effective_num_cores=4,
        world_size=4,
    )
    runtime = replace(
        runtime,
        runtime_plan=replace(
            runtime.runtime_plan,
            environment=environment,
            allocations=(allocation,),
        ),
    )

    _apply_worker_runtime_environment(runtime)

    assert os.environ["NEURON_RT_VISIBLE_CORES"] == "4,5,6,7"
    assert os.environ["NEURON_RT_NUM_CORES"] == "4"


@pytest.mark.parametrize(
    "runtime,error",
    [
        (
            lambda runtime: replace(
                runtime,
                runtime_plan=replace(
                    runtime.runtime_plan,
                    allocations=(
                        runtime.runtime_plan.allocations[0],
                        replace(
                            runtime.runtime_plan.allocations[0],
                            allocation_id="extra-worker",
                        ),
                    ),
                ),
            ),
            "exactly one allocation",
        ),
        (
            lambda runtime: replace(
                runtime,
                runtime_plan=replace(
                    runtime.runtime_plan,
                    environment=replace(
                        runtime.runtime_plan.environment,
                        available_core_ids=(),
                    ),
                ),
            ),
            "exceeds available cores",
        ),
    ],
)
def test_worker_environment_rejects_invalid_allocation_before_mutation(monkeypatch, runtime, error):
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "9")

    with pytest.raises(ValueError, match=error):
        _apply_worker_runtime_environment(runtime(_runtime()))

    assert os.environ["NEURON_RT_NUM_CORES"] == "9"


def test_resident_worker_engine_recovers_after_inflight_timeout():
    async def _run():
        engine = ResidentWorkerServingEngine(
            runtime=_runtime(),
            orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
            config=ResidentWorkerConfig(request_timeout=0.1, worker_restart_timeout=5),
        )
        await engine.start()
        try:
            with pytest.raises(DiffletServingError) as exc:
                await engine.generate(_request("req-timeout", "sleep:1"))
            assert exc.value.code == "request_timeout"

            for _ in range(50):
                if engine.ready:
                    break
                await asyncio.sleep(0.05)

            out = await engine.generate(_request("req-2", "hello"))
        finally:
            await engine.shutdown()
        return out

    output = asyncio.run(_run())

    assert output.data == b"hello:64x64"


def test_resident_worker_engine_restarts_when_cancel_is_not_acknowledged():
    async def _run():
        engine = ResidentWorkerServingEngine(
            runtime=_runtime(),
            orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
            config=ResidentWorkerConfig(
                request_timeout=0.2,
                worker_cancel_timeout=0.02,
                worker_restart_timeout=5,
            ),
        )
        await engine.start()
        try:
            with pytest.raises(DiffletServingError) as exc:
                await engine.generate(_request("req-no-ack", "ignore-cancel:1.0"))
            assert exc.value.code == "request_timeout"

            for _ in range(100):
                if engine.ready:
                    break
                await asyncio.sleep(0.05)

            out = await engine.generate(_request("req-after-restart", "hello"))
        finally:
            await engine.shutdown()
        return out

    output = asyncio.run(_run())

    assert output.data == b"hello:64x64"


def test_resident_worker_engine_ready_reflects_idle_worker_death():
    async def _run():
        engine = ResidentWorkerServingEngine(
            runtime=_runtime(),
            orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
            config=ResidentWorkerConfig(request_timeout=5, worker_restart_timeout=5),
        )
        await engine.start()
        try:
            assert engine.ready
            assert engine.healthy
            engine._worker.terminate()
            return engine.ready, engine.healthy
        finally:
            await engine.shutdown()

    ready, healthy = asyncio.run(_run())

    assert ready is False
    assert healthy is False


def test_resident_worker_engine_health_stays_ok_during_recovery():
    engine = ResidentWorkerServingEngine(
        runtime=_runtime(),
        orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
    )
    engine._recovering = True
    engine._worker.mark_error()

    assert engine.ready is False
    assert engine.healthy is True

    engine._unrecoverable = True

    assert engine.healthy is False


def test_queue_wait_uses_request_timeout_when_deadline_expires():
    async def _run():
        engine = ResidentWorkerServingEngine(
            runtime=_runtime(),
            orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
            config=ResidentWorkerConfig(
                queue_timeout=5,
                request_timeout=0.01,
                worker_restart_timeout=5,
            ),
        )
        engine._worker.ready = True
        engine._worker.healthy = True
        await engine._run_lock.acquire()
        try:
            with pytest.raises(DiffletServingError) as exc:
                await engine.generate(_request("req-deadline", "hello"))
            return exc.value.code
        finally:
            engine._run_lock.release()

    assert asyncio.run(_run()) == "request_timeout"


def test_cancelled_queued_caller_does_not_later_acquire_run_lock():
    async def _run():
        engine = ResidentWorkerServingEngine(
            runtime=_runtime(),
            orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
        )
        engine._worker.ready = True
        engine._worker.healthy = True
        await engine._run_lock.acquire()
        queued = asyncio.create_task(engine.generate(_request("cancelled-queue", "hello")))
        while engine._pending < 1:
            await asyncio.sleep(0)

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        engine._run_lock.release()

        await asyncio.wait_for(engine._run_lock.acquire(), timeout=0.1)
        engine._run_lock.release()
        return engine._pending

    assert asyncio.run(_run()) == 0


def test_closed_engine_recovery_does_not_restart_worker(monkeypatch):
    async def _run():
        engine = ResidentWorkerServingEngine(
            runtime=_runtime(),
            orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
        )
        starts = []
        monkeypatch.setattr(engine._worker, "start", lambda: starts.append(True))
        engine._closed = True
        engine._draining = True
        engine._recovering = True
        run_task = asyncio.create_task(asyncio.sleep(0))

        await engine._recover_worker(run_task, reason="engine_unavailable")
        await run_task
        return starts

    assert asyncio.run(_run()) == []


def test_shutdown_terminalizes_running_and_queued_requests_as_draining():
    async def _run():
        engine = ResidentWorkerServingEngine(
            runtime=_runtime(),
            orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
            config=ResidentWorkerConfig(request_timeout=5, worker_restart_timeout=5),
        )
        await engine.start()
        running = asyncio.create_task(engine.generate(_request("running", "sleep:0.3")))
        while engine._pending < 1:
            await asyncio.sleep(0)
        queued = asyncio.create_task(engine.generate(_request("queued", "hello")))
        while engine._pending < 2:
            await asyncio.sleep(0)

        await engine.shutdown()
        results = await asyncio.gather(running, queued, return_exceptions=True)
        return results

    results = asyncio.run(_run())

    assert all(isinstance(result, DiffletServingError) for result in results)
    assert [result.code for result in results] == ["engine_draining", "engine_draining"]


def test_unknown_worker_error_is_sanitized_before_ipc(caplog):
    caplog.set_level(logging.ERROR, logger="difflet.serving.engines.resident_worker")

    reply = _reply_from_error(RuntimeError("secret cache path /private/models"), request_id="r1")

    assert reply["code"] == "internal_error"
    assert reply["message"] == "Internal model execution error"
    assert "/private/models" not in str(reply)
    assert "/private/models" in caplog.text


def test_non_allowlisted_worker_reply_is_sanitized_defensively():
    error = _error_from_reply(
        {
            "status_code": 503,
            "code": "engine_unavailable",
            "message": "secret backend detail",
            "error_type": "server_error",
        }
    )

    assert error.code == "internal_error"
    assert error.message == "Internal model execution error"


@pytest.mark.parametrize("interval", [0, 4.99, 120.01, float("nan"), float("inf"), -float("inf")])
def test_resident_worker_config_rejects_invalid_heartbeat_interval(interval):
    with pytest.raises(ValueError, match="between 5 and 120 seconds inclusive"):
        ResidentWorkerConfig(worker_heartbeat_interval=interval)


@pytest.mark.parametrize("interval", [5, 120])
def test_resident_worker_config_accepts_heartbeat_boundaries(interval):
    assert (
        ResidentWorkerConfig(worker_heartbeat_interval=interval).worker_heartbeat_interval
        == interval
    )


def test_worker_heartbeat_logs_independently_of_reply_queue(caplog):
    async def _run():
        engine = ResidentWorkerServingEngine(
            runtime=_runtime(),
            orchestrator_factory="tests.unit.serving.fake_worker:FakeServingOrchestrator",
            config=ResidentWorkerConfig(
                request_timeout=10,
                worker_restart_timeout=10,
                worker_heartbeat_interval=5,
            ),
        )
        await engine.start()
        try:
            running = asyncio.create_task(
                engine.generate(_request("heartbeat-running", "sleep:5.1"))
            )
            while engine._pending < 1:
                await asyncio.sleep(0)
            queued = asyncio.create_task(engine.generate(_request("heartbeat-queued", "hello")))
            while engine._pending < 2:
                await asyncio.sleep(0)
            await asyncio.gather(running, queued)
        finally:
            await engine.shutdown()

    caplog.set_level(logging.INFO, logger="difflet.serving.engines.resident_worker")
    asyncio.run(_run())

    heartbeats = [
        record.worker_heartbeat for record in caplog.records if hasattr(record, "worker_heartbeat")
    ]
    assert heartbeats
    assert any(event["state"] == "busy" for event in heartbeats)
    assert any(event["stage"] == "pipeline" for event in heartbeats)
    busy_with_queue = next(
        event for event in heartbeats if event["state"] == "busy" and event["queued_requests"] == 1
    )
    assert busy_with_queue["running_requests"] == 1
    assert busy_with_queue["pending_requests"] == 2
    assert busy_with_queue["request_capacity"] == 9
    assert "running_requests=1 queued_requests=1 pending_requests=2 capacity=9" in caplog.text
