from __future__ import annotations

import asyncio

from difflet.serving.types import DiffletGenerateOutput


class FakeServingOrchestrator:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.active_profile = None

    def load(self, profile) -> None:
        self.active_profile = profile

    def smoke(self) -> None:
        if self.active_profile is None:
            raise RuntimeError("not loaded")

    async def generate(self, request, context):
        context.cancellation.throw_if_cancelled()
        if request.prompt.startswith("sleep:"):
            remaining = float(request.prompt.split(":", 1)[1])
            while remaining > 0:
                context.cancellation.throw_if_cancelled()
                step = min(remaining, 0.01)
                await asyncio.sleep(step)
                remaining -= step
        if request.prompt.startswith("ignore-cancel:"):
            await asyncio.sleep(float(request.prompt.split(":", 1)[1]))
        return DiffletGenerateOutput(
            data=f"{request.prompt}:{request.height}x{request.width}".encode(),
            mime_type="image/png",
            output_format="png",
        )

    def shutdown(self) -> None:
        self.active_profile = None
