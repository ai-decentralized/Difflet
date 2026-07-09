"""Serving adapter protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.types import (
    DiffletCompileSpec,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    DiffletStageSpec,
    ServingProfile,
    WorkerRequestContext,
)


class ServingArtifactPreparer(Protocol):
    model_id: str
    model_type: str

    def resolve_model_path(self, *, download_policy: DownloadPolicy) -> Path: ...

    def stage_specs(self, profile: ServingProfile) -> tuple[DiffletStageSpec, ...]: ...

    def compile_plan(self, profile: ServingProfile) -> tuple[DiffletCompileSpec, ...]: ...

    def ensure_artifacts(self, profile: ServingProfile, policy: CompilePolicy) -> None: ...


class ServingRequestValidator(Protocol):
    def validate(self, request: DiffletGenerateRequest, profile: ServingProfile) -> None: ...


class NoopServingRequestValidator:
    def validate(self, request: DiffletGenerateRequest, profile: ServingProfile) -> None:
        return None


class ServingModelOrchestrator(Protocol):
    model_id: str
    model_type: str
    active_profile: ServingProfile | None

    def load(self, profile: ServingProfile) -> None: ...

    def smoke(self) -> None: ...

    async def generate(
        self,
        request: DiffletGenerateRequest,
        context: WorkerRequestContext,
    ) -> DiffletGenerateOutput: ...

    def shutdown(self) -> None: ...
