from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict

from difflet.serving.engines.stage_pipeline import ValidatedStageRunner, stage_result
from difflet.serving.types import (
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    FluxFinalPayload,
    FluxInitialPayload,
    StageExecutionResult,
    StageInvocation,
)


class _FakePipelineRunner:
    def __init__(self, adapter: "FakeServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[FluxInitialPayload],
    ) -> StageExecutionResult[FluxFinalPayload]:
        started = time.monotonic()
        request = invocation.request
        context = invocation.context
        context.cancellation.throw_if_cancelled()
        if request.prompt == "runtime-env":
            values = [
                os.environ.get("NEURON_RT_VISIBLE_CORES", ""),
                os.environ.get("NEURON_RT_NUM_CORES", ""),
                os.environ.get("NEURON_RT_VIRTUAL_CORE_SIZE", ""),
                os.environ.get("NEURON_LOGICAL_NC_CONFIG", ""),
                os.environ.get("WORLD_SIZE", ""),
                os.environ.get("LOCAL_WORLD_SIZE", ""),
                os.environ.get("RANK", ""),
                os.environ.get("LOCAL_RANK", ""),
            ]
            data = ":".join(values).encode()
        elif request.prompt.startswith("sleep:"):
            remaining = float(request.prompt.split(":", 1)[1])
            while remaining > 0:
                context.cancellation.throw_if_cancelled()
                step = min(remaining, 0.01)
                await asyncio.sleep(step)
                remaining -= step
            data = f"{request.prompt}:{request.height}x{request.width}".encode()
        elif request.prompt.startswith("ignore-cancel:"):
            await asyncio.sleep(float(request.prompt.split(":", 1)[1]))
            data = f"{request.prompt}:{request.height}x{request.width}".encode()
        elif request.prompt.startswith("error:"):
            raise RuntimeError(request.prompt.split(":", 1)[1])
        else:
            data = f"{request.prompt}:{request.height}x{request.width}".encode()
        return stage_result(
            FluxFinalPayload(DiffletGenerateOutput(data, "image/png", "png")),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        self.adapter.active_runtime = None


class FakeServingStageAdapter:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.active_runtime = None

    async def create_loaded_runners(self, runtime):
        self.active_runtime = runtime
        return OrderedDict(
            (
                (
                    "pipeline",
                    ValidatedStageRunner(
                        _FakePipelineRunner(self), FluxInitialPayload, FluxFinalPayload
                    ),
                ),
            )
        )

    def smoke_request(self) -> DiffletGenerateRequest:
        if self.active_runtime is None:
            raise RuntimeError("not loaded")
        profile = self.active_runtime.profile
        return DiffletGenerateRequest(
            "startup-smoke",
            profile.model_id,
            "smoke",
            profile.height,
            profile.width,
            1,
            1.0,
            0,
        )

    def initial_payload(self, request):
        return FluxInitialPayload()

    def finalize(self, payload):
        if type(payload) is not FluxFinalPayload:
            raise TypeError("invalid fake final payload")
        return payload.output

    def validate_smoke_output(self, output):
        if not output.data:
            raise RuntimeError("empty smoke output")

    def reset_request_state(self, outcome):
        return None

    async def shutdown(self) -> None:
        self.active_runtime = None


FakeServingOrchestrator = FakeServingStageAdapter
