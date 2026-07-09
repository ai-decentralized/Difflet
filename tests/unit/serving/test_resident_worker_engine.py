from __future__ import annotations

import asyncio

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.errors import DiffletServingError
from difflet.serving.engines.resident_worker import ResidentWorkerConfig, ResidentWorkerServingEngine
from difflet.serving.types import DiffletGenerateRequest, ServingProfile


def _profile() -> ServingProfile:
    return ServingProfile(
        model_id="fake/model",
        model_type="fake",
        height=64,
        width=64,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=1),
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
            profile=_profile(),
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


def test_resident_worker_engine_recovers_after_inflight_timeout():
    async def _run():
        engine = ResidentWorkerServingEngine(
            profile=_profile(),
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
            profile=_profile(),
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
            profile=_profile(),
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
        profile=_profile(),
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
            profile=_profile(),
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
