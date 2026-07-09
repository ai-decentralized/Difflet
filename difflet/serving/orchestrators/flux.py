"""Flux serving adapter."""

from __future__ import annotations

import io
from pathlib import Path

from difflet.common.orchestrators import flux as flux_common
from difflet.registry import resolve_model
from difflet.serving.errors import prompt_too_long
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.types import (
    DiffletCompileSpec,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    DiffletStageSpec,
    ServingProfile,
    WorkerRequestContext,
)

_HF_MODEL_ID = flux_common.HF_MODEL_ID
_MODEL_TYPE = flux_common.MODEL_TYPE
_MAX_SEQUENCE_LENGTH = flux_common.MAX_SEQUENCE_LENGTH


class FluxServingArtifactPreparer:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision

    def resolve_model_path(self, *, download_policy: DownloadPolicy) -> Path:
        from difflet.pipeline.path_resolver import resolve_model_path

        entry = resolve_model(self.model_id, model_type=_MODEL_TYPE)
        print(f"[difflet serve] resolving Flux weights for {self.model_id}")
        path = resolve_model_path(
            self.model_id,
            revision=self.revision,
            local_files_only=download_policy == DownloadPolicy.NEVER,
            allow_patterns=entry.download_patterns,
        )
        print(f"[difflet serve] Flux weights ready at {path}")
        return Path(path)

    def stage_specs(self, profile: ServingProfile) -> tuple[DiffletStageSpec, ...]:
        return (
            DiffletStageSpec(
                stage_id="pipeline",
                role="pipeline",
                num_cores=profile.parallel.world_size,
                output_keys=("image",),
                final_output=True,
            ),
        )

    def compile_plan(self, profile: ServingProfile) -> tuple[DiffletCompileSpec, ...]:
        pipe = flux_common.build_pipeline(
            self.model_id,
            profile,
            load=False,
            skip_compile=True,
        )
        return (
            DiffletCompileSpec(
                stage_id="pipeline",
                artifact_path=str(pipe.compiled_path),
                required=True,
            ),
        )

    def ensure_artifacts(self, profile: ServingProfile, policy: CompilePolicy) -> None:
        force = policy == CompilePolicy.FORCE
        skip_compile = policy == CompilePolicy.NEVER
        print(
            "[difflet serve] checking Flux AOT artifacts "
            f"(policy={policy.value}, force={force})"
        )
        pipe = self._build_pipeline(
            profile,
            load=False,
            force_compile=force,
            skip_compile=skip_compile,
        )
        if not flux_common.pipeline_artifacts_ready(pipe):
            raise RuntimeError(f"missing Flux compiled artifacts at {pipe.compiled_path}")
        print("[difflet serve] Flux AOT artifacts ready")

    def _build_pipeline(
        self,
        profile: ServingProfile,
        *,
        load: bool,
        force_compile: bool = False,
        skip_compile: bool = False,
    ):
        return flux_common.build_pipeline(
            self.model_id,
            profile,
            force_compile=force_compile,
            skip_compile=skip_compile,
            load=load,
        )


class FluxServingRequestValidator:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision
        self._tokenizer = None

    def validate(self, request: DiffletGenerateRequest, profile: ServingProfile) -> None:
        encoded = self._tokenizer_for_profile(profile)(
            request.prompt,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        if int(encoded.input_ids.shape[1]) > _MAX_SEQUENCE_LENGTH:
            raise prompt_too_long(f"Flux prompt exceeds text bucket {_MAX_SEQUENCE_LENGTH}")

    def _tokenizer_for_profile(self, profile: ServingProfile):
        if self._tokenizer is None:
            from difflet.pipeline.path_resolver import resolve_model_path
            from transformers import AutoTokenizer

            model_dir = resolve_model_path(
                self.model_id,
                revision=profile.revision,
                local_files_only=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer_2"))
        return self._tokenizer


class FluxServingOrchestrator:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.active_profile: ServingProfile | None = None
        self.pipe = None

    def load(self, profile: ServingProfile) -> None:
        print(f"[difflet serve] loading Flux worker profile {profile}")
        self.active_profile = profile
        self.pipe = FluxServingArtifactPreparer(self.model_id, profile.revision)._build_pipeline(
            profile,
            load=True,
            skip_compile=True,
        )
        print("[difflet serve] Flux worker loaded")

    def smoke(self) -> None:
        if self.pipe is None:
            raise RuntimeError("Flux worker is not loaded")
        print("[difflet serve] Flux smoke passed")

    async def generate(
        self,
        request: DiffletGenerateRequest,
        context: WorkerRequestContext,
    ) -> DiffletGenerateOutput:
        import torch

        if self.pipe is None:
            raise RuntimeError("Flux worker is not loaded")
        context.cancellation.throw_if_cancelled()
        output = self.pipe(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            height=request.height,
            width=request.width,
            guidance_scale=request.guidance_scale,
            generator=torch.Generator().manual_seed(request.seed),
        )
        context.cancellation.throw_if_cancelled()
        image = output.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return DiffletGenerateOutput(data=buf.getvalue(), mime_type="image/png", output_format="png")

    def shutdown(self) -> None:
        self.pipe = None
        self.active_profile = None
