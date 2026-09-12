"""Flux serving adapter."""

from __future__ import annotations

import io
import logging
import time
from collections import OrderedDict
from pathlib import Path

from difflet.common.orchestrators import flux as flux_common
from difflet.registry import resolve_model
from difflet.serving.artifact_manager import ArtifactPublishTarget, ImmutableArtifactManager
from difflet.serving.errors import profile_mismatch, prompt_too_long
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators.base import (
    request_uses_teacache,
    resolve_available_neuron_core_ids,
    resolve_hf_model_source,
    validate_guidance_scale,
)
from difflet.serving.engines.stage_pipeline import (
    ErasedStageRunner,
    ValidatedStageRunner,
    require_exact_payload,
    stage_result,
)
from difflet.serving.types import (
    ArtifactSet,
    DistributedProcessEnvironment,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    FluxFinalPayload,
    FluxInitialPayload,
    ParallelTopology,
    ResolvedRuntimeBundle,
    RuntimeEnvironment,
    RuntimePlan,
    ServingProfile,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
    StageRuntimeSpec,
    WorkerAllocationSpec,
)

_HF_MODEL_ID = flux_common.HF_MODEL_ID
_MODEL_TYPE = flux_common.MODEL_TYPE
_MAX_SEQUENCE_LENGTH = flux_common.MAX_SEQUENCE_LENGTH
_MAX_GUIDANCE_SCALE = 20.0

logger = logging.getLogger(__name__)


def _backend_is_tpu() -> bool:
    from difflet.backends.registry import current_backend

    return current_backend() == "tpu"


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
        if _backend_is_tpu():
            # Eager stages, no compile artifacts (see the LTX-2 / Wan adapters).
            print("[difflet serve] tpu backend: eager stages, no compile artifacts")
            pipeline = _pipeline_definition()
            return ResolvedRuntimeBundle(
                profile=profile,
                source=source,
                pipeline_definition=pipeline,
                runtime_plan=_runtime_plan(profile, pipeline, ()),
                compile_specs=(),
                artifacts=ArtifactSet(()),
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

        bindings = tuple(_prepare_binding(spec) for spec in specs)
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

    def preload(self) -> None:
        self._tokenizer_for_runtime()

    def validate(self, request: DiffletGenerateRequest) -> None:
        validate_guidance_scale(request, maximum=_MAX_GUIDANCE_SCALE)
        profile = self.runtime.profile
        # Strict membership in the compiled bucket set: the NxD router only
        # accepts exactly-compiled shapes, so anything else is rejected here
        # (with the allowed set) instead of surfacing a runtime ValueError.
        if (request.height, request.width, None) not in profile.shape_set():
            allowed = [f"{h}x{w}" for h, w, _ in profile.canonical_shapes()]
            raise profile_mismatch(
                f"request shape {request.height}x{request.width} is not in the "
                f"Flux serving profile's compiled shape set {allowed}"
            )
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


class FluxPipelineRunner:
    def __init__(self, pipe, profile: ServingProfile) -> None:
        self.pipe = pipe
        self.profile = profile

    async def execute(
        self,
        invocation: StageInvocation[FluxInitialPayload],
    ) -> StageExecutionResult[FluxFinalPayload]:
        import torch

        started = time.monotonic()
        request = invocation.request
        context = invocation.context
        context.cancellation.throw_if_cancelled()
        use_teacache = request_uses_teacache(self.profile, request.num_inference_steps)
        if self.profile.teacache_speedup is not None and not use_teacache:
            logger.info(
                "Flux request uses baseline inference fallback_reason=step_mismatch "
                "request_steps=%s calibration_steps=%s",
                request.num_inference_steps,
                getattr(self.profile.teacache_calibration_data, "num_steps", None),
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
        return stage_result(
            FluxFinalPayload(
                DiffletGenerateOutput(
                    data=buf.getvalue(), mime_type="image/png", output_format="png"
                )
            ),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        self.pipe = None


class FluxServingStageAdapter:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.active_runtime: ResolvedRuntimeBundle | None = None
        self.active_profile: ServingProfile | None = None
        self._untransferred_pipe = None

    async def create_loaded_runners(
        self,
        runtime: ResolvedRuntimeBundle,
    ) -> OrderedDict[str, ErasedStageRunner]:
        profile = runtime.profile
        print(f"[difflet serve] loading Flux worker profile {profile}")
        self.active_runtime = runtime
        self.active_profile = profile
        if _backend_is_tpu():
            import torch

            from difflet.models.flux.entry import create_flux_application

            app = create_flux_application(
                model_path=runtime.source.pinned_model_path,
                parallel=profile.parallel,
                dtype=torch.bfloat16,
                shape=profile.shape_dict(),
                backend="tpu",
                teacache_cadence=profile.teacache_cadence,
                teacache_online_delta_alpha=profile.teacache_online_delta,
            )
            app.load_eager()
            runner = ValidatedStageRunner(
                FluxPipelineRunner(app, profile), FluxInitialPayload, FluxFinalPayload
            )
            print("[difflet serve] Flux worker loaded (tpu)", flush=True)
            return OrderedDict((("pipeline", runner),))
        binding = runtime.artifacts.require("pipeline")
        spec = runtime.require_compile_spec("pipeline")
        manager = ImmutableArtifactManager(profile.cache_dir or Path.home() / ".cache" / "difflet")
        manager.validate_binding(
            binding,
            validate_payload=lambda path: flux_common.validate_compiled_artifact(
                runtime.source, profile, spec, path
            ),
        )
        pipe = flux_common.build_pipeline(
            self.model_id,
            profile,
            load=True,
            skip_compile=True,
            model_path_override=runtime.source.pinned_model_path,
            resolved_source_id=runtime.source.resolved_source_id,
            compiled_path_override=str(binding.path),
        )
        print("[difflet serve] Flux worker loaded")
        self._untransferred_pipe = pipe
        runner = ValidatedStageRunner(
            FluxPipelineRunner(pipe, profile),
            FluxInitialPayload,
            FluxFinalPayload,
        )
        self._untransferred_pipe = None
        return OrderedDict((("pipeline", runner),))

    def initial_payload(self, request: DiffletGenerateRequest) -> FluxInitialPayload:
        return FluxInitialPayload()

    def finalize(self, payload: StagePayload) -> DiffletGenerateOutput:
        return require_exact_payload(
            payload,
            FluxFinalPayload,
            boundary="Flux final payload",
        ).output

    def smoke_request(self) -> DiffletGenerateRequest:
        if self.active_profile is None:
            raise RuntimeError("Flux serving profile is not loaded")
        profile = self.active_profile
        print("[difflet serve] running Flux generation smoke")
        return DiffletGenerateRequest(
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

    def validate_smoke_output(self, output: DiffletGenerateOutput) -> None:
        if self.active_profile is None:
            raise RuntimeError("Flux serving profile is not loaded")
        profile = self.active_profile
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

    def reset_request_state(self, outcome: str) -> None:
        return None

    async def shutdown(self) -> None:
        self._untransferred_pipe = None
        self.active_profile = None
        self.active_runtime = None


def _pipeline_definition():
    from difflet.common.registry.flux import serving_metadata

    return serving_metadata().pipeline_definition


def _runtime_plan(profile: ServingProfile, pipeline, specs) -> RuntimePlan:
    world_size = profile.world_size
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment(
        available_core_ids=resolve_available_neuron_core_ids(required_num_cores=world_size),
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
    # No specs means the TPU backend (eager, no artifacts): the stage carries
    # placement="tpu" and no artifact id, as in the other TPU adapters.
    tpu = not specs
    spec = None if tpu else specs[0]
    stages = (
        StageRuntimeSpec(
            stage_id=pipeline.stages[0].stage_id,
            allocation_id=allocation.allocation_id,
            topology=ParallelTopology(
                tp_degree=profile.parallel.tp_degree,
                cp_degree=profile.parallel.cp_degree,
                world_size=world_size,
            ),
            artifact_id=None if tpu else spec.artifact_id,
            placement="tpu" if tpu else "neuron",
        ),
    )
    return RuntimePlan(
        mode="resident",
        profile_identity="" if tpu else spec.identity.digest,
        environment=environment,
        allocations=(allocation,),
        stages=stages,
    )
