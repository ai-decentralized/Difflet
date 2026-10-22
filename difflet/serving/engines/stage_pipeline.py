"""Generic sequential stage execution for the resident serving worker."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Generic, Protocol

from difflet.serving.types import (
    DiffletGenerateRequest,
    GenerateOutput,
    InputPayloadT,
    OutputPayloadT,
    ResolvedRuntimeBundle,
    StageExecutionMetadata,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
    WorkerRequestContext,
)

logger = logging.getLogger(__name__)


class StagePayloadTypeError(TypeError):
    """A stage received or returned a payload outside its declared contract."""


class StageShutdownError(RuntimeError):
    """One or more stage cleanup operations failed."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = tuple(errors)
        super().__init__(f"{len(errors)} stage cleanup operation(s) failed")


class StageRunner(Protocol[InputPayloadT, OutputPayloadT]):
    async def execute(
        self,
        invocation: StageInvocation[InputPayloadT],
    ) -> StageExecutionResult[OutputPayloadT]: ...

    async def shutdown(self) -> None: ...


class ErasedStageRunner(Protocol):
    @property
    def input_type(self) -> type[StagePayload]: ...

    @property
    def output_type(self) -> type[StagePayload]: ...

    async def execute(
        self,
        invocation: StageInvocation[StagePayload],
    ) -> StageExecutionResult[StagePayload]: ...

    async def shutdown(self) -> None: ...


class ValidatedStageRunner(Generic[InputPayloadT, OutputPayloadT]):
    """Erase generic runner types while enforcing their exact runtime contract."""

    def __init__(
        self,
        inner: StageRunner[InputPayloadT, OutputPayloadT],
        input_type: type[InputPayloadT],
        output_type: type[OutputPayloadT],
    ) -> None:
        self.inner = inner
        self._input_type = input_type
        self._output_type = output_type

    @property
    def input_type(self) -> type[StagePayload]:
        return self._input_type

    @property
    def output_type(self) -> type[StagePayload]:
        return self._output_type

    async def execute(
        self,
        invocation: StageInvocation[StagePayload],
    ) -> StageExecutionResult[StagePayload]:
        if (
            not isinstance(invocation.input, self._input_type)
            or type(invocation.input) is not self._input_type
        ):
            raise StagePayloadTypeError(
                f"stage input must be exactly {self._input_type.__name__}; "
                f"got {type(invocation.input).__name__}"
            )
        typed_invocation = StageInvocation(
            request=invocation.request,
            stage=invocation.stage,
            input=invocation.input,
            context=invocation.context,
        )
        result = await self.inner.execute(typed_invocation)
        if (
            not isinstance(result.output, self._output_type)
            or type(result.output) is not self._output_type
        ):
            raise StagePayloadTypeError(
                f"stage output must be exactly {self._output_type.__name__}; "
                f"got {type(result.output).__name__}"
            )
        return StageExecutionResult(output=result.output, metadata=result.metadata)

    async def shutdown(self) -> None:
        await self.inner.shutdown()


class InProcessStageExecutor:
    """P0 executor that invokes all runners sequentially in the current process."""

    def __init__(self, runners: Mapping[str, ErasedStageRunner]) -> None:
        self.runners = dict(runners)

    async def execute(
        self,
        invocation: StageInvocation[StagePayload],
    ) -> StageExecutionResult[StagePayload]:
        if not isinstance(invocation.input, StagePayload):
            raise StagePayloadTypeError(
                f"stage {invocation.stage.stage_id!r} input must inherit StagePayload"
            )
        try:
            runner = self.runners[invocation.stage.stage_id]
        except KeyError as exc:
            raise StagePayloadTypeError(
                f"missing runner for stage {invocation.stage.stage_id!r}"
            ) from exc
        result = await runner.execute(invocation)
        if not isinstance(result.output, StagePayload):
            raise StagePayloadTypeError(
                f"stage {invocation.stage.stage_id!r} output must inherit StagePayload"
            )
        return result

    async def shutdown(self) -> None:
        errors: list[BaseException] = []
        for runner in reversed(tuple(self.runners.values())):
            try:
                await runner.shutdown()
            except BaseException as exc:
                errors.append(exc)
        self.runners.clear()
        if errors:
            raise StageShutdownError(errors)


class ServingStageAdapter(Protocol):
    model_id: str
    model_type: str

    async def create_loaded_runners(
        self,
        runtime: ResolvedRuntimeBundle,
    ) -> Mapping[str, ErasedStageRunner]: ...

    def initial_payload(self, request: DiffletGenerateRequest) -> StagePayload: ...

    def finalize(self, payload: StagePayload) -> GenerateOutput: ...

    def smoke_request(self) -> DiffletGenerateRequest: ...

    def validate_smoke_output(self, output: GenerateOutput) -> None: ...

    def reset_request_state(self, outcome: str) -> None: ...

    async def shutdown(self) -> None: ...


class StagePipelineEngine:
    """Execute one immutable pipeline definition through a selected stage executor."""

    def __init__(
        self,
        *,
        runtime: ResolvedRuntimeBundle,
        adapter: ServingStageAdapter,
    ) -> None:
        self.runtime = runtime
        self.adapter = adapter
        self.executor: InProcessStageExecutor | None = None
        self._closed = False

    async def start(self) -> None:
        try:
            runners = await self.adapter.create_loaded_runners(self.runtime)
            expected = tuple(stage.stage_id for stage in self.runtime.pipeline_definition.stages)
            actual = tuple(runners)
            if actual != expected:
                raise StagePayloadTypeError(
                    f"runner IDs/order {actual!r} do not match pipeline {expected!r}"
                )
            self.executor = InProcessStageExecutor(runners)
            request = self.adapter.smoke_request()
            output = await self.generate(
                request,
                WorkerRequestContext.with_timeout(request.request_id, 300.0),
            )
            self.adapter.validate_smoke_output(output)
        except BaseException:
            await self._cleanup_after_failure()
            raise

    async def generate(
        self,
        request: DiffletGenerateRequest,
        context: WorkerRequestContext,
    ) -> GenerateOutput:
        if self.executor is None or self._closed:
            raise RuntimeError("stage pipeline engine is not ready")
        self.adapter.reset_request_state("before_request")
        try:
            payload = self.adapter.initial_payload(request)
            if not isinstance(payload, StagePayload):
                raise StagePayloadTypeError("initial payload must inherit StagePayload")
            stages = self.runtime.pipeline_definition.stages
            for index, stage in enumerate(stages):
                context.cancellation.throw_if_cancelled()
                context.report_stage(stage.stage_id)
                logger.info(
                    "stage.start request_id=%s model=%s stage=%s",
                    request.request_id,
                    request.model,
                    stage.stage_id,
                )
                result = await self.executor.execute(
                    StageInvocation(
                        request=request,
                        stage=stage,
                        input=payload,
                        context=context,
                    )
                )
                context.cancellation.throw_if_cancelled()
                logger.info(
                    "stage.complete request_id=%s model=%s stage=%s duration=%.3f",
                    request.request_id,
                    request.model,
                    stage.stage_id,
                    result.metadata.finished_monotonic - result.metadata.started_monotonic,
                )
                if stage.final_output:
                    if index != len(stages) - 1:
                        raise StagePayloadTypeError("final stage must be last")
                    output = self.adapter.finalize(result.output)
                    self.adapter.reset_request_state("completed")
                    return output
                payload = result.output
            raise StagePayloadTypeError("pipeline has no final output stage")
        except BaseException:
            self.adapter.reset_request_state("error")
            raise

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        if self.executor is not None:
            try:
                await self.executor.shutdown()
            except BaseException as exc:
                errors.extend(exc.errors if isinstance(exc, StageShutdownError) else (exc,))
            self.executor = None
        try:
            await self.adapter.shutdown()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise StageShutdownError(errors)

    async def _cleanup_after_failure(self) -> None:
        try:
            await self.shutdown()
        except BaseException:
            # Startup already failed; the worker process is discarded by its parent.
            pass


def stage_result(
    output: OutputPayloadT,
    *,
    started_monotonic: float,
) -> StageExecutionResult[OutputPayloadT]:
    return StageExecutionResult(
        output=output,
        metadata=StageExecutionMetadata(
            started_monotonic=started_monotonic,
            finished_monotonic=time.monotonic(),
        ),
    )


def require_exact_payload(
    value: StagePayload,
    expected_type: type[OutputPayloadT],
    *,
    boundary: str,
) -> OutputPayloadT:
    if not isinstance(value, expected_type) or type(value) is not expected_type:
        raise StagePayloadTypeError(
            f"{boundary} must be exactly {expected_type.__name__}; got {type(value).__name__}"
        )
    return value
