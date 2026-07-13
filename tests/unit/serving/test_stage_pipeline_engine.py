from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
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
