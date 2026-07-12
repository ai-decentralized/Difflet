from __future__ import annotations

import asyncio
import os

from difflet.serving.types import DiffletGenerateOutput


class FakeServingOrchestrator:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.active_runtime = None

    def load(self, runtime) -> None:
        self.active_runtime = runtime

    def smoke(self) -> None:
        if self.active_runtime is None:
            raise RuntimeError("not loaded")

    async def generate(self, request, context):
        context.cancellation.throw_if_cancelled()
        context.report_stage("pipeline")
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
            return DiffletGenerateOutput(
                data=":".join(values).encode(),
                mime_type="image/png",
                output_format="png",
            )
        if request.prompt.startswith("sleep:"):
            remaining = float(request.prompt.split(":", 1)[1])
            while remaining > 0:
                context.cancellation.throw_if_cancelled()
                step = min(remaining, 0.01)
                await asyncio.sleep(step)
                remaining -= step
        if request.prompt.startswith("ignore-cancel:"):
            await asyncio.sleep(float(request.prompt.split(":", 1)[1]))
        if request.prompt.startswith("error:"):
            raise RuntimeError(request.prompt.split(":", 1)[1])
        return DiffletGenerateOutput(
            data=f"{request.prompt}:{request.height}x{request.width}".encode(),
            mime_type="image/png",
            output_format="png",
        )

    def shutdown(self) -> None:
        self.active_runtime = None
