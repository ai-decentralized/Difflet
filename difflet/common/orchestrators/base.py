"""Common orchestration protocols used by serving adapters."""

from __future__ import annotations

from typing import Protocol

from difflet.serving.types import DiffletGenerateOutput, DiffletGenerateRequest
from difflet.serving.types import ServingProfile, WorkerRequestContext


class CommonGenerateHandler(Protocol):
    active_profile: ServingProfile

    async def generate(
        self,
        request: DiffletGenerateRequest,
        context: WorkerRequestContext,
    ) -> DiffletGenerateOutput: ...

