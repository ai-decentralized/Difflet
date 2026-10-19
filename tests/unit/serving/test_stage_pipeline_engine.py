from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.engines.stage_observability import DeviceMemorySample, record_stage_load
from difflet.serving.engines.stage_pipeline import (
    InProcessStageExecutor,
    StagePayloadTypeError,
    StagePipelineEngine,
    StageShutdownError,
    ValidatedStageRunner,
    stage_result,
)
from difflet.serving.types import (
    ArtifactBinding,
    ArtifactSet,
    CompileArtifactIdentity,
    DiffletCompileSpec,
    DiffletGenerateOutput,
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
    StageExecutionMetadata,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
    StageRuntimeSpec,
    WorkerAllocationSpec,
    WorkerRequestContext,
)


@dataclass(frozen=True, slots=True)
class _Initial(StagePayload):
    value: str


@dataclass(frozen=True, slots=True)
class _Middle(StagePayload):
    value: str


@dataclass(frozen=True, slots=True)
class _Final(StagePayload):
    output: DiffletGenerateOutput


class _InitialSubclass(_Initial):
    pass


class _MiddleSubclass(_Middle):
    pass


class _Runner:
    def __init__(
        self,
        output,
        *,
        calls=None,
        shutdown_error: BaseException | None = None,
        shutdown_label: str = "shutdown",
    ):
        self.output = output
        self.calls = calls if calls is not None else []
        self.shutdown_error = shutdown_error
        self.shutdown_label = shutdown_label

    async def execute(self, invocation):
        self.calls.append(invocation.stage.stage_id)
        return stage_result(self.output, started_monotonic=time.monotonic())

    async def shutdown(self):
        self.calls.append(self.shutdown_label)
        if self.shutdown_error is not None:
            raise self.shutdown_error


def _request(request_id: str = "request") -> DiffletGenerateRequest:
    return DiffletGenerateRequest(request_id, "fake/model", "prompt", 64, 64, 1, 1.0, 0)


def _context(request_id: str = "request") -> WorkerRequestContext:
    return WorkerRequestContext.with_timeout(request_id, 1.0)


def _runtime() -> ResolvedRuntimeBundle:
    profile = ServingProfile(
        model_id="fake/model",
        model_type="fake",
        height=64,
        width=64,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=1),
    )
    pipeline = PipelineDefinition(
        "fake",
        (
            StageDefinition("first", "extracted", "prompt_encoder", runner_factory="x"),
            StageDefinition("last", "extracted", "decoder", final_output=True, runner_factory="x"),
        ),
    )
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment((0,), None, None, None, distributed, distributed)
    allocation = WorkerAllocationSpec("worker", 1, 1, 1)
    topology = ParallelTopology(1, 1, 1)
    identities = {
        stage.stage_id: CompileArtifactIdentity.from_cache_inputs({"stage": stage.stage_id})
        for stage in pipeline.stages
    }
    specs = tuple(
        DiffletCompileSpec(stage_id, stage_id, identities[stage_id]) for stage_id in identities
    )
    bindings = tuple(
        ArtifactBinding(
            stage_id,
            Path(f"/tmp/{stage_id}"),
            Path(f"/tmp/{stage_id}/manifest"),
            identity,
            "g1",
            "0" * 64,
        )
        for stage_id, identity in identities.items()
    )
    return ResolvedRuntimeBundle(
        profile,
        ResolvedModelSource("hf_snapshot", "fake/model", None, "/tmp/model", "a" * 40),
        pipeline,
        RuntimePlan(
            "resident",
            "profile",
            environment,
            (allocation,),
            tuple(
                StageRuntimeSpec(stage.stage_id, "worker", topology, stage.stage_id)
                for stage in pipeline.stages
            ),
        ),
        specs,
        ArtifactSet(bindings),
    )


def test_validated_runner_rejects_wrong_and_subclass_input():
    runner = ValidatedStageRunner(_Runner(_Middle("ok")), _Initial, _Middle)
    stage = _runtime().pipeline_definition.stages[0]

    for payload in (_Middle("wrong"), _InitialSubclass("subclass")):
        with pytest.raises(StagePayloadTypeError, match="stage input"):
            asyncio.run(runner.execute(StageInvocation(_request(), stage, payload, _context())))


def test_validated_runner_rejects_non_payload_and_subclass_output():
    stage = _runtime().pipeline_definition.stages[0]
    for output in (object(), _MiddleSubclass("subclass")):
        runner = ValidatedStageRunner(_Runner(output), _Initial, _Middle)
        with pytest.raises(StagePayloadTypeError, match="stage output"):
            asyncio.run(
                runner.execute(StageInvocation(_request(), stage, _Initial("ok"), _context()))
            )


def test_executor_rejects_non_nominal_initial_payload():
    runner = ValidatedStageRunner(_Runner(_Middle("ok")), _Initial, _Middle)
    executor = InProcessStageExecutor(OrderedDict((("first", runner),)))
    stage = _runtime().pipeline_definition.stages[0]

    with pytest.raises(StagePayloadTypeError, match="must inherit StagePayload"):
        asyncio.run(executor.execute(StageInvocation(_request(), stage, object(), _context())))


def test_executor_shutdown_is_exhaustive_and_reverse_order():
    calls = []
    first = ValidatedStageRunner(
        _Runner(_Middle("a"), calls=calls, shutdown_label="first_shutdown"),
        _Initial,
        _Middle,
    )
    second = ValidatedStageRunner(
        _Runner(
            _Final(DiffletGenerateOutput(b"x", "image/png")),
            calls=calls,
            shutdown_error=RuntimeError("boom"),
            shutdown_label="last_shutdown",
        ),
        _Middle,
        _Final,
    )
    executor = InProcessStageExecutor(OrderedDict((("first", first), ("last", second))))

    with pytest.raises(StageShutdownError) as exc:
        asyncio.run(executor.shutdown())

    assert calls == ["last_shutdown", "first_shutdown"]
    assert len(exc.value.errors) == 1


class _Adapter:
    model_id = "fake/model"
    model_type = "fake"

    def __init__(self, *, wrong_final: bool = False, invalid_initial: bool = False):
        self.calls = []
        self.wrong_final = wrong_final
        self.invalid_initial = invalid_initial

    async def create_loaded_runners(self, runtime):
        return OrderedDict(
            (
                (
                    "first",
                    ValidatedStageRunner(
                        _Runner(_Middle("middle"), calls=self.calls), _Initial, _Middle
                    ),
                ),
                (
                    "last",
                    ValidatedStageRunner(
                        _Runner(
                            _Final(DiffletGenerateOutput(b"ok", "image/png")),
                            calls=self.calls,
                        ),
                        _Middle,
                        _Final,
                    ),
                ),
            )
        )

    def initial_payload(self, request):
        if self.invalid_initial:
            return object()
        return _Initial(request.prompt)

    def finalize(self, payload):
        expected = _Middle if self.wrong_final else _Final
        if type(payload) is not expected:
            raise StagePayloadTypeError("wrong final payload")
        return payload.output

    def smoke_request(self):
        return _request("smoke")

    def validate_smoke_output(self, output):
        assert output.data == b"ok"

    def reset_request_state(self, outcome):
        self.calls.append(outcome)

    async def shutdown(self):
        self.calls.append("adapter_shutdown")


def test_stage_pipeline_engine_traverses_ordered_stages_and_finalizes():
    adapter = _Adapter()
    engine = StagePipelineEngine(runtime=_runtime(), adapter=adapter)

    async def _run():
        await engine.start()
        output = await engine.generate(_request(), _context())
        await engine.shutdown()
        return output

    output = asyncio.run(_run())

    assert output.data == b"ok"
    assert adapter.calls[:4] == ["before_request", "first", "last", "completed"]
    assert adapter.calls[4:8] == ["before_request", "first", "last", "completed"]
    assert adapter.calls[-3:] == ["shutdown", "shutdown", "adapter_shutdown"]


def test_stage_pipeline_engine_rejects_wrong_final_payload():
    adapter = _Adapter(wrong_final=True)
    engine = StagePipelineEngine(runtime=_runtime(), adapter=adapter)

    with pytest.raises(StagePayloadTypeError, match="wrong final payload"):
        asyncio.run(engine.start())


def test_stage_pipeline_engine_rejects_non_payload_initial_value():
    adapter = _Adapter(invalid_initial=True)
    engine = StagePipelineEngine(runtime=_runtime(), adapter=adapter)

    with pytest.raises(StagePayloadTypeError, match="initial payload"):
        asyncio.run(engine.start())

    assert "error" in adapter.calls


class _SlowRunner(_Runner):
    """A runner whose metadata reports a fixed duration, so timing assertions are exact."""

    def __init__(self, output, seconds: float, *, calls=None):
        super().__init__(output, calls=calls)
        self.seconds = float(seconds)

    async def execute(self, invocation):
        self.calls.append(invocation.stage.stage_id)
        started = time.monotonic()
        return StageExecutionResult(
            output=self.output,
            metadata=StageExecutionMetadata(
                started_monotonic=started,
                finished_monotonic=started + self.seconds,
            ),
        )


class _TimedAdapter(_Adapter):
    """``_Adapter`` whose two stages report 1 s and 9 s, and which marks its load boundaries."""

    def __init__(self, *, mark_loads: bool = True, fail_in_last: bool = False):
        super().__init__()
        self.mark_loads = mark_loads
        self.fail_in_last = fail_in_last

    async def create_loaded_runners(self, runtime):
        first = ValidatedStageRunner(
            _SlowRunner(_Middle("middle"), 1.0, calls=self.calls), _Initial, _Middle
        )
        last = ValidatedStageRunner(
            (
                _ExplodingRunner(calls=self.calls)
                if self.fail_in_last
                else _SlowRunner(
                    _Final(DiffletGenerateOutput(b"ok", "image/png")), 9.0, calls=self.calls
                )
            ),
            _Middle,
            _Final,
        )
        if self.mark_loads:
            with record_stage_load("first"):
                pass
            with record_stage_load("last"):
                pass
        return OrderedDict((("first", first), ("last", last)))


class _ExplodingRunner(_Runner):
    def __init__(self, *, calls=None):
        super().__init__(None, calls=calls)

    async def execute(self, invocation):
        self.calls.append(invocation.stage.stage_id)
        raise RuntimeError("stage blew up")


class _StubProbe:
    def __init__(self, readings):
        self.name = "stub"
        self._readings = list(readings)

    def sample(self):
        if not self._readings:
            return None
        return DeviceMemorySample(int(self._readings.pop(0)), "process")

    @property
    def unavailable_reason(self):
        return "stub exhausted"


def _engine(adapter, probe=None):
    return StagePipelineEngine(
        runtime=_runtime(),
        adapter=adapter,
        device_memory_probe=probe or _StubProbe([0] * 32),
    )


def test_engine_records_each_stage_wall_time_for_every_request():
    adapter = _TimedAdapter()
    engine = _engine(adapter)

    async def _run():
        await engine.start()
        await engine.generate(_request("r1"), _context("r1"))

    asyncio.run(_run())

    breakdown = engine.last_stage_breakdown
    assert breakdown is not None
    assert breakdown.request_id == "r1"
    assert breakdown.outcome == "completed"
    assert breakdown.stage_seconds == {"first": 1.0, "last": 9.0}
    assert breakdown.total_seconds >= 0.0


def test_engine_logs_the_stage_breakdown_once_per_request(caplog):
    adapter = _TimedAdapter()
    engine = _engine(adapter)

    with caplog.at_level(logging.INFO, logger="difflet.serving.engines.stage_pipeline"):

        async def _run():
            await engine.start()
            await engine.generate(_request("r1"), _context("r1"))

        asyncio.run(_run())

    breakdowns = [
        record for record in caplog.records if record.getMessage().startswith("stage.breakdown")
    ]
    # One for the startup smoke, one for the request.
    assert [record.stage_breakdown["request_id"] for record in breakdowns] == ["smoke", "r1"]
    message = breakdowns[-1].getMessage()
    assert "first=1.000s(" in message and "last=9.000s(" in message
    assert breakdowns[-1].stage_breakdown["stages"] == [
        {"stage_id": "first", "seconds": 1.0},
        {"stage_id": "last", "seconds": 9.0},
    ]


def test_engine_keeps_the_partial_breakdown_when_a_stage_fails():
    adapter = _TimedAdapter(fail_in_last=True)
    engine = _engine(adapter)

    with pytest.raises(RuntimeError, match="stage blew up"):
        asyncio.run(engine.start())

    breakdown = engine.last_stage_breakdown
    assert breakdown is not None
    assert breakdown.outcome == "error"
    # The stage that ran is recorded; the one that raised is not timed.
    assert list(breakdown.stage_seconds) == ["first"]


def test_engine_reports_per_stage_residency_from_the_adapter_boundaries():
    # baseline 0; first stage 0 -> 4; last stage 4 -> 10; final 10.
    probe = _StubProbe([0, 0, 4, 4, 10, 10])
    adapter = _TimedAdapter()
    engine = _engine(adapter, probe)

    asyncio.run(engine.start())

    report = engine.stage_load_report
    assert report is not None
    assert report.attribution == "per_stage"
    assert {stage.stage_id: stage.device_bytes for stage in report.stages} == {
        "first": 4,
        "last": 6,
    }
    assert report.total_device_bytes == 10
    assert report.unattributed_device_bytes == 0


def test_engine_logs_the_total_when_no_adapter_boundary_exists(caplog):
    adapter = _TimedAdapter(mark_loads=False)
    engine = _engine(adapter, _StubProbe([0, 42]))

    with caplog.at_level(logging.INFO, logger="difflet.serving.engines.stage_pipeline"):
        asyncio.run(engine.start())

    report = engine.stage_load_report
    assert report is not None
    assert report.attribution == "total_only"
    assert report.total_device_bytes == 42
    summary = next(
        record for record in caplog.records if record.getMessage().startswith("stage.load_summary")
    )
    assert "attribution=total_only" in summary.getMessage()
    assert "no stage marked its load boundary" in summary.getMessage()
    assert summary.stage_load_summary["total_device_bytes"] == 42


def test_engine_says_unavailable_when_the_backend_cannot_report_memory(caplog):
    adapter = _TimedAdapter()
    engine = _engine(adapter, _StubProbe([]))

    with caplog.at_level(logging.INFO, logger="difflet.serving.engines.stage_pipeline"):
        asyncio.run(engine.start())

    report = engine.stage_load_report
    assert report is not None
    assert report.attribution == "unavailable"
    assert report.total_device_bytes is None
    summary = next(
        record for record in caplog.records if record.getMessage().startswith("stage.load_summary")
    )
    assert "attribution=unavailable" in summary.getMessage()
    assert "stub exhausted" in summary.getMessage()
    assert "device_memory=unknown" in next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("stage.load ")
    )
