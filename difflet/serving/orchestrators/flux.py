"""Flux serving adapter."""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

from difflet.common.orchestrators import flux as flux_common
from difflet.registry import resolve_model
from difflet.serving.artifact_manager import ArtifactPublishTarget, ImmutableArtifactManager
from difflet.serving.errors import prompt_too_long
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators.base import request_uses_teacache, resolve_hf_model_source
from difflet.serving.types import (
    ArtifactSet,
    DistributedProcessEnvironment,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    ParallelTopology,
    ResolvedRuntimeBundle,
    RuntimeEnvironment,
    RuntimePlan,
    ServingProfile,
    StageRuntimeSpec,
    WorkerAllocationSpec,
    WorkerRequestContext,
)

_HF_MODEL_ID = flux_common.HF_MODEL_ID
_MODEL_TYPE = flux_common.MODEL_TYPE
_MAX_SEQUENCE_LENGTH = flux_common.MAX_SEQUENCE_LENGTH

logger = logging.getLogger(__name__)


class FluxServingArtifactPreparer:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision

    def prepare_runtime(
        self,
        profile: ServingProfile,
        *,
        download_policy: DownloadPolicy,
        compile_policy: CompilePolicy,
    ) -> ResolvedRuntimeBundle:
        entry = resolve_model(self.model_id, model_type=_MODEL_TYPE)
        print(f"[difflet serve] resolving Flux weights for {self.model_id}")
        source = resolve_hf_model_source(
            self.model_id,
            revision=self.revision,
            download_policy=download_policy,
            allow_patterns=entry.download_patterns,
        )
        specs = flux_common.build_compile_plan(source, profile)
        manager = ImmutableArtifactManager(profile.cache_dir or Path.home() / ".cache" / "difflet")

        def _prepare_binding(spec):
            def _compile_artifact(target: ArtifactPublishTarget) -> None:
                flux_common.compile_serving_artifact(source, profile, spec, target)

            def _validate_payload(path: Path) -> None:
                flux_common.validate_compiled_artifact(source, profile, spec, path)

            return manager.prepare(
                model_type=self.model_type,
                artifact_id=spec.artifact_id,
                identity=spec.identity,
                policy=compile_policy,
                compile_artifact=_compile_artifact,
                validate_payload=_validate_payload,
            )

        bindings = tuple(
            _prepare_binding(spec)
            for spec in specs
        )
        pipeline = _pipeline_definition()
        runtime_plan = _runtime_plan(profile, pipeline, specs)
        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=pipeline,
            runtime_plan=runtime_plan,
            compile_specs=specs,
            artifacts=ArtifactSet(bindings),
        )


class FluxServingRequestValidator:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, runtime: ResolvedRuntimeBundle) -> None:
        self.model_id = runtime.profile.model_id
        self.runtime = runtime
        self._tokenizer = None

    def validate(self, request: DiffletGenerateRequest) -> None:
        encoded = self._tokenizer_for_runtime()(
            request.prompt,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        if int(encoded.input_ids.shape[1]) > _MAX_SEQUENCE_LENGTH:
            raise prompt_too_long(f"Flux prompt exceeds text bucket {_MAX_SEQUENCE_LENGTH}")

    def _tokenizer_for_runtime(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_dir = self.runtime.source.pinned_model_path
            self._tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer_2"))
        return self._tokenizer


class FluxServingOrchestrator:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.active_runtime: ResolvedRuntimeBundle | None = None
        self.active_profile: ServingProfile | None = None
        self.pipe = None

    def load(self, runtime: ResolvedRuntimeBundle) -> None:
        profile = runtime.profile
        print(f"[difflet serve] loading Flux worker profile {profile}")
        self.active_runtime = runtime
        self.active_profile = profile
        binding = runtime.artifacts.require("pipeline")
        spec = runtime.require_compile_spec("pipeline")
        manager = ImmutableArtifactManager(profile.cache_dir or Path.home() / ".cache" / "difflet")
        manager.validate_binding(
            binding,
            validate_payload=lambda path: flux_common.validate_compiled_artifact(
                runtime.source, profile, spec, path
            ),
        )
        self.pipe = flux_common.build_pipeline(
            self.model_id,
            profile,
            load=True,
            skip_compile=True,
            model_path_override=runtime.source.pinned_model_path,
            resolved_source_id=runtime.source.resolved_source_id,
            compiled_path_override=str(binding.path),
        )
        print("[difflet serve] Flux worker loaded")

    def smoke(self) -> None:
        if self.pipe is None:
            raise RuntimeError("Flux worker is not loaded")
        if self.active_profile is None:
            raise RuntimeError("Flux serving profile is not loaded")
        profile = self.active_profile
        print("[difflet serve] running Flux generation smoke")
        request = DiffletGenerateRequest(
            request_id="startup-smoke",
            model=self.model_id,
            prompt="a small red square",
            height=profile.height,
            width=profile.width,
            num_inference_steps=(
                profile.teacache_calibration_data.num_steps
                if profile.teacache_calibration_data is not None
                else 4
            ),
            guidance_scale=1.0,
            seed=0,
        )
        output = asyncio.run(
            self.generate(request, WorkerRequestContext.with_timeout("startup-smoke", 300.0))
        )
        if not output.data:
            raise RuntimeError("Flux smoke produced empty output")
        from PIL import Image

        with Image.open(io.BytesIO(output.data)) as image:
            if image.size != (profile.width, profile.height):
                raise RuntimeError(
                    "Flux smoke output shape mismatch: "
                    f"expected {profile.width}x{profile.height}, "
                    f"got {image.width}x{image.height}"
                )
        print("[difflet serve] Flux generation smoke passed")

    async def generate(
        self,
        request: DiffletGenerateRequest,
        context: WorkerRequestContext,
    ) -> DiffletGenerateOutput:
        import torch

        if self.pipe is None:
            raise RuntimeError("Flux worker is not loaded")
        if self.active_profile is None:
            raise RuntimeError("Flux serving profile is not loaded")
        context.cancellation.throw_if_cancelled()
        context.report_stage("pipeline")
        use_teacache = request_uses_teacache(
            self.active_profile,
            request.num_inference_steps,
        )
        if self.active_profile.teacache_speedup is not None and not use_teacache:
            logger.info(
                "Flux request uses baseline inference fallback_reason=step_mismatch "
                "request_steps=%s calibration_steps=%s",
                request.num_inference_steps,
                getattr(self.active_profile.teacache_calibration_data, "num_steps", None),
            )
        output = self.pipe(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            height=request.height,
            width=request.width,
            guidance_scale=request.guidance_scale,
            generator=torch.Generator().manual_seed(request.seed),
            teacache_enabled=use_teacache,
        )
        context.cancellation.throw_if_cancelled()
        image = output.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return DiffletGenerateOutput(
            data=buf.getvalue(), mime_type="image/png", output_format="png"
        )

    def shutdown(self) -> None:
        self.pipe = None
        self.active_profile = None
        self.active_runtime = None


def _pipeline_definition():
    from difflet.common.registry.flux import serving_metadata

    return serving_metadata().pipeline_definition


def _runtime_plan(profile: ServingProfile, pipeline, specs) -> RuntimePlan:
    world_size = profile.world_size
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment(
        available_core_ids=tuple(range(world_size)),
        num_cores_override=None,
        virtual_core_size_override=None,
        logical_nc_config_override=None,
        inherited_distributed=distributed,
        child_distributed=distributed,
    )
    allocation = WorkerAllocationSpec(
        allocation_id="flux-resident",
        requested_num_cores=world_size,
        effective_num_cores=world_size,
        world_size=world_size,
    )
    spec = specs[0]
    stages = (
        StageRuntimeSpec(
            stage_id=pipeline.stages[0].stage_id,
            allocation_id=allocation.allocation_id,
            topology=ParallelTopology(
                tp_degree=profile.parallel.tp_degree,
                cp_degree=profile.parallel.cp_degree,
                world_size=world_size,
            ),
            artifact_id=spec.artifact_id,
        ),
    )
    return RuntimePlan(
        mode="resident",
        profile_identity=spec.identity.digest,
        environment=environment,
        allocations=(allocation,),
        stages=stages,
    )
