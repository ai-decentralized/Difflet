"""``difflet serve`` for every model on the generic GPU flow.

Each served model is one ``DiffletPipeline`` resident in one worker process on one GPU, behind
a single opaque ``pipeline`` stage that calls ``app.generate()`` per request with the same
kwargs ``difflet run`` uses (``difflet/backends/gpu/cli.py``). The model's ``ModelSpec`` supplies
the request contract (defaults, guidance rules, prompt budgets, refused options) and the output
kind; the engine, the HTTP layer and the Trainium adapters are untouched.

The registry constructs these classes by import path: the preparer and the adapter with the
model id, the validator with the resolved bundle. Each finds its model's spec from that.

Why the plan claims core ``0``: the resident worker validates its allocation generically (one
allocation, positive core count, enough ``available_core_ids``) and exports
``NEURON_RT_VISIBLE_CORES`` from it. Those "cores" are abstract device slots to the engine; a GPU
worker never loads the Neuron runtime, so the one GPU is slot 0 and the variable is inert
(``--dp-degree N`` rewrites the slot per replica).

Importing this module must not import torch: the serving parent process resolves the preparer
and validator factories itself.
"""

from __future__ import annotations

import io
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from difflet.backends.gpu.policy import (
    cfg_parallel_does_not_apply,
    gpu_platform,
    is_set,
    unsupported_parallel,
)
from difflet.common.registry.base import ServingModelMetadata
from difflet.models.spec import ModelSpec, gpu_model_spec
from difflet.platforms.base import PARALLEL_MODE_FLAGS
from difflet.serving.engines.stage_pipeline import (
    ErasedStageRunner,
    ValidatedStageRunner,
    require_exact_payload,
    stage_result,
)
from difflet.serving.errors import (
    feature_not_supported,
    invalid_extra_body,
    profile_mismatch,
    prompt_too_long,
)
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators.base import resolve_hf_model_source, validate_guidance_scale
from difflet.serving.types import (
    ArtifactSet,
    CompileArtifactIdentity,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    DistributedProcessEnvironment,
    FileBackedGenerateOutput,
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
    VideoGenerateOptions,
    WorkerAllocationSpec,
)

STAGE_ID = "pipeline"
_MODULE = "difflet.backends.gpu.serving"


@dataclass(frozen=True, slots=True)
class GpuInitialPayload(StagePayload):
    pass


@dataclass(frozen=True, slots=True)
class GpuImagePayload(StagePayload):
    output: DiffletGenerateOutput


@dataclass(frozen=True, slots=True)
class GpuVideoPayload(StagePayload):
    output: FileBackedGenerateOutput


# --------------------------------------------------------------------- metadata


def gpu_serving_metadata(spec: ModelSpec) -> ServingModelMetadata:
    """The serving metadata of ``spec`` on the GPU flow: one resident process, one stage."""
    video = spec.output == "video"
    guidance = (
        spec.default_guidance_scale
        if spec.default_guidance_scale is not None
        else spec.serving_noop_guidance_scale
    )
    return ServingModelMetadata(
        model_type=spec.model_type,
        checkpoint_ids=spec.served_checkpoint_ids,
        output_modality=spec.output,
        output_mime_type="video/mp4" if video else "image/png",
        chat_content_type="video_url" if video else "image_url",
        default_steps=spec.default_steps,
        default_guidance_scale=float(guidance),
        pipeline_definition=PipelineDefinition(
            model_type=spec.model_type,
            stages=(
                StageDefinition(
                    stage_id=STAGE_ID,
                    kind="opaque_pipeline",
                    role="pipeline",
                    output_keys=("output",),
                    final_output=True,
                ),
            ),
        ),
        artifact_preparer_factory=f"{_MODULE}:GpuServingArtifactPreparer",
        orchestrator_factory=f"{_MODULE}:GpuServingStageAdapter",
        request_validator_factory=f"{_MODULE}:GpuServingRequestValidator",
        default_fps=spec.default_fps,
        default_host_vae=False,
    )


def _spec_for_model_id(model_id: str) -> ModelSpec:
    from difflet.registry import resolve_model

    return gpu_model_spec(resolve_model(model_id).name)


def _tag(spec: ModelSpec) -> str:
    return f"[{spec.model_type}/{gpu_platform().name} serve]"


def _refuse(spec: ModelSpec, what: str) -> None:
    raise feature_not_supported(
        f"{_tag(spec)} {what} is not supported on the {gpu_platform().display_name} backend yet"
    )


def refuse_unsupported_profile(profile: ServingProfile, spec: ModelSpec) -> None:
    """Raise ``feature_not_supported`` for every profile the GPU flow does not serve: one
    request shape, one card per replica, and the model's own refusals.

    ``build_serving_profile`` already refuses ``--cfg-parallel`` and the TeaCache modes it
    cannot serve for every model; these checks are the worker's backstop for a bundle that did
    not come through it."""
    if profile.shapes:
        _refuse(spec, "--shapes (bucketed request shapes)")
    parallel = profile.parallel
    for mode in unsupported_parallel(spec, parallel, gpu_platform()):
        if mode == "cfg":
            raise feature_not_supported(f"{_tag(spec)} {cfg_parallel_does_not_apply(spec)}")
        _refuse(spec, PARALLEL_MODE_FLAGS[mode])
    if parallel.cfg_parallel_enabled:
        _refuse(spec, "--cfg-parallel (one CFG branch per GPU, a `difflet run` feature)")
    for attribute, label in spec.refused_serving_options:
        if is_set(getattr(profile, attribute, None)):
            _refuse(spec, label)


def resident_runtime_plan(
    profile: ServingProfile, source: ResolvedModelSource, spec: ModelSpec
) -> RuntimePlan:
    """The resident plan for one GPU: one allocation (slot 0), one eager stage placed on the
    GPU platform.

    The profile identity hashes the backend, model, pinned source, shape, tp and dtype, and
    ``num_frames`` for a video profile only."""
    platform = gpu_platform()
    inputs = {
        "backend": platform.name,
        "model_type": profile.model_type,
        "source_id": source.resolved_source_id,
        "height": int(profile.height),
        "width": int(profile.width),
        "tp_degree": int(profile.parallel.tp_degree),
        "dtype": profile.dtype,
    }
    if profile.num_frames is not None:
        inputs["num_frames"] = int(profile.num_frames)
    identity = CompileArtifactIdentity.from_cache_inputs(inputs)
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment(
        available_core_ids=(0,),
        num_cores_override=None,
        virtual_core_size_override=None,
        logical_nc_config_override=None,
        inherited_distributed=distributed,
        child_distributed=distributed,
    )
    allocation_id = f"{spec.short_name}-{platform.name}"
    allocation = WorkerAllocationSpec(
        allocation_id=allocation_id,
        requested_num_cores=1,
        effective_num_cores=1,
        world_size=1,
    )
    stage = StageRuntimeSpec(
        stage_id=STAGE_ID,
        allocation_id=allocation_id,
        topology=ParallelTopology(tp_degree=1, cp_degree=1, world_size=1),
        artifact_id=None,
        placement=platform.name,
    )
    return RuntimePlan(
        mode="resident",
        profile_identity=identity.digest,
        environment=environment,
        allocations=(allocation,),
        stages=(stage,),
    )


# --------------------------------------------------------------------- preparer


class GpuServingArtifactPreparer:
    def __init__(self, model_id: str, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision
        self.spec = _spec_for_model_id(model_id)
        self.model_type = self.spec.model_type

    def prepare_runtime(
        self,
        profile: ServingProfile,
        *,
        download_policy: DownloadPolicy,
        compile_policy: CompilePolicy,
    ) -> ResolvedRuntimeBundle:
        spec = self.spec
        refuse_unsupported_profile(profile, spec)
        print(f"[difflet serve] resolving {spec.display_name} weights for {self.model_id}")
        source = resolve_hf_model_source(
            self.model_id, revision=self.revision, download_policy=download_policy
        )
        # No compile phase: the flow is eager, and DiffletPipeline writes its manifest-only
        # artifact inside the worker at load. An empty spec/binding pair keeps
        # ResolvedRuntimeBundle's invariant true.
        print(
            f"[difflet serve] {gpu_platform().name} backend: eager resident pipeline, "
            "no compile artifacts"
        )
        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=gpu_serving_metadata(spec).pipeline_definition,
            runtime_plan=resident_runtime_plan(profile, source, spec),
            compile_specs=(),
            artifacts=ArtifactSet(()),
            adapter_config={"compile_policy": compile_policy},
        )


# -------------------------------------------------------------------- validator


def _video(request: DiffletGenerateRequest, spec: ModelSpec) -> VideoGenerateOptions:
    if request.output_format != "mp4":
        raise invalid_extra_body(f"{spec.display_name} serving writes MP4 only")
    if request.video is None:
        raise invalid_extra_body(
            f"a {spec.display_name} request needs its video options (num_frames, fps)"
        )
    return request.video


class GpuServingRequestValidator:
    """The request contract, enforced in the server process before the worker."""

    def __init__(self, runtime: ResolvedRuntimeBundle) -> None:
        self.runtime = runtime
        self.model_id = runtime.profile.model_id
        self.spec = gpu_model_spec(runtime.profile.model_type)
        self.model_type = self.spec.model_type
        self._tokenizers: dict[str, Any] = {}

    def preload(self) -> None:
        for budget in self.spec.prompt_budgets:
            self._tokenizer(budget.tokenizer_subdir)

    def validate(self, request: DiffletGenerateRequest) -> None:
        spec = self.spec
        profile = self.runtime.profile
        video = _video(request, spec) if spec.output == "video" else None
        self._validate_guidance(request)
        if video is None:
            if (request.height, request.width) != (profile.height, profile.width):
                raise profile_mismatch(
                    f"request shape {request.height}x{request.width} is not the resident "
                    f"profile shape {profile.height}x{profile.width}"
                )
        else:
            requested = (request.height, request.width, video.num_frames)
            resident = (profile.height, profile.width, profile.num_frames)
            if requested != resident:
                raise profile_mismatch(
                    "request video shape {}x{}x{} is not the resident profile shape "
                    "{}x{}x{}".format(*requested, *resident)
                )
            if video.fps != profile.output_fps:
                raise profile_mismatch(
                    f"request fps {video.fps} is not the resident profile's {profile.output_fps}"
                )
            g2 = video.guidance_scale_2
            if g2 is not None and not 0 <= float(g2) <= spec.max_guidance_scale:
                raise invalid_extra_body(
                    f"guidance_scale_2 must satisfy 0 <= value <= {spec.max_guidance_scale:g}"
                )
            for field, why in spec.refused_video_fields:
                if getattr(video, field) is not None:
                    raise feature_not_supported(
                        f"{_tag(spec)} {field} is not taken by {spec.display_name}: {why}"
                    )
        self._validate_prompts(request, video)

    def _validate_guidance(self, request: DiffletGenerateRequest) -> None:
        spec = self.spec
        if spec.guidance != "none":
            validate_guidance_scale(request, maximum=spec.max_guidance_scale)
            return
        if float(request.guidance_scale) != spec.serving_noop_guidance_scale:
            raise feature_not_supported(
                f"guidance_scale has no effect on {spec.display_name} "
                f"({spec.no_guidance_reason}); omit it"
            )

    def _validate_prompts(self, request: DiffletGenerateRequest, video) -> None:
        for budget in self.spec.prompt_budgets:
            texts = [("prompt", request.prompt)]
            if budget.applies_to_negative_prompt and video is not None:
                texts.append(("negative_prompt", video.negative_prompt))
            for name, text in texts:
                if text is None:
                    continue
                if budget.template is not None:
                    text = budget.template.format(text)
                encoded = self._tokenizer(budget.tokenizer_subdir)(
                    text, padding=False, truncation=False, return_tensors="pt"
                )
                if len(encoded.input_ids[0]) - budget.drop_prefix_tokens > budget.max_tokens:
                    raise prompt_too_long(budget.too_long.format(name=name))

    def _tokenizer(self, subdir: str):
        if subdir not in self._tokenizers:
            from transformers import AutoTokenizer

            model_dir = self.runtime.source.pinned_model_path
            self._tokenizers[subdir] = AutoTokenizer.from_pretrained(str(Path(model_dir) / subdir))
        return self._tokenizers[subdir]


# --------------------------------------------------------------------- adapter


class GpuPipelineRunner:
    """The one stage: prompt in, PNG or MP4 out, through the resident ``DiffletPipeline``."""

    def __init__(self, adapter: "GpuServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(self, invocation: StageInvocation[GpuInitialPayload]) -> StageExecutionResult:
        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        if self.adapter.spec.output != "video":
            output = self.adapter.generate_image(invocation.request)
            return stage_result(GpuImagePayload(output=output), started_monotonic=started)
        from difflet.serving.models._common import encode_video_tensor

        frames = self.adapter.generate_frames(invocation.request)
        invocation.context.cancellation.throw_if_cancelled()
        output = encode_video_tensor(
            frames, invocation.request, layout="BCTHW", value_range="minus_one_to_one"
        )
        return stage_result(GpuVideoPayload(output=output), started_monotonic=started)

    async def shutdown(self) -> None:
        self.adapter.pipe = None


class GpuServingStageAdapter:
    """``ServingStageAdapter`` for a model on the GPU flow: the model stays resident."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.spec = _spec_for_model_id(model_id)
        self.model_type = self.spec.model_type
        self.active_runtime: ResolvedRuntimeBundle | None = None
        self.active_profile: ServingProfile | None = None
        self.pipe: Any = None
        self._smoke = None

    @property
    def _final_payload(self) -> type:
        return GpuVideoPayload if self.spec.output == "video" else GpuImagePayload

    async def create_loaded_runners(
        self,
        runtime: ResolvedRuntimeBundle,
    ) -> OrderedDict[str, ErasedStageRunner]:
        spec = self.spec
        profile = runtime.profile
        refuse_unsupported_profile(profile, spec)
        # Worker-side imports: the parent process never loads torch for this adapter.
        from difflet.backends.gpu.cli import gpu_pipeline_kwargs
        from difflet.pipeline.difflet_pipeline import DiffletPipeline

        config = runtime.adapter_config or {}
        print(f"[difflet serve] loading {spec.display_name} on {gpu_platform().display_name}")
        self.active_runtime = runtime
        self.active_profile = profile
        self.pipe = DiffletPipeline.from_pretrained(
            self.model_id,
            **gpu_pipeline_kwargs(
                spec,
                tp_degree=profile.parallel.tp_degree,
                height=profile.height,
                width=profile.width,
                num_frames=profile.num_frames,
                cache_dir=profile.cache_dir,
                # The commit the parent pinned (resolve_hf_model_source), not the user's
                # ref: a worker restart must load the snapshot the bundle and the
                # validator's tokenizer describe, even if refs/main moved meanwhile.
                revision=runtime.source.resolved_source_id,
                force_compile=config.get("compile_policy") is CompilePolicy.FORCE,
                load=True,
                torch_compile=profile.torch_compile,
            ),
        )
        print(f"[difflet serve] {spec.display_name} pipeline resident")
        return OrderedDict(
            (
                (
                    STAGE_ID,
                    ValidatedStageRunner(
                        GpuPipelineRunner(self), GpuInitialPayload, self._final_payload
                    ),
                ),
            )
        )

    def generate_image(self, request: DiffletGenerateRequest) -> DiffletGenerateOutput:
        """``app.generate`` with the CLI's arguments; no ``guidance_scale`` for a model that
        takes none (its application refuses it)."""
        if self.pipe is None:
            raise RuntimeError(f"{self.spec.display_name} pipeline is not loaded")
        kwargs: dict[str, Any] = dict(
            num_inference_steps=int(request.num_inference_steps),
            seed=int(request.seed),
            output_type="pil",
        )
        if self.spec.guidance != "none":
            kwargs["guidance_scale"] = float(request.guidance_scale)
        image = self.pipe.app.generate(request.prompt, **kwargs)[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return DiffletGenerateOutput(
            data=buf.getvalue(), mime_type="image/png", output_format="png"
        )

    def generate_frames(self, request: DiffletGenerateRequest):
        """``app.generate`` with the CLI's arguments, plus ``guidance_scale_2`` and
        ``negative_prompt`` when the request carries them."""
        if self.pipe is None or self.active_profile is None:
            raise RuntimeError(f"{self.spec.display_name} pipeline is not loaded")
        import torch

        video = _video(request, self.spec)
        kwargs: dict[str, Any] = dict(
            num_inference_steps=int(request.num_inference_steps),
            guidance_scale=float(request.guidance_scale),
            seed=int(request.seed),
            num_frames=int(self.active_profile.num_frames),
            output_type="pt",
        )
        if video.guidance_scale_2 is not None:
            kwargs["guidance_scale_2"] = float(video.guidance_scale_2)
        if video.negative_prompt is not None:
            kwargs["negative_prompt"] = video.negative_prompt
        frames = self.pipe.app.generate(request.prompt, **kwargs)
        return frames.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)

    def initial_payload(self, request: DiffletGenerateRequest) -> GpuInitialPayload:
        return GpuInitialPayload()

    def finalize(self, payload: StagePayload):
        return require_exact_payload(
            payload, self._final_payload, boundary=f"{self.spec.display_name} final payload"
        ).output

    def smoke_request(self) -> DiffletGenerateRequest:
        """A cheap startup request (``spec.smoke_steps``) that proves load plus one generate.
        For a model with two experts the steps straddle the expert boundary, so a broken
        second expert fails startup, not a request."""
        spec = self.spec
        if self.pipe is None or self.active_profile is None:
            raise RuntimeError(f"{spec.display_name} load did not leave a resident pipeline")
        profile = self.active_profile
        guidance = (
            spec.default_guidance_scale
            if spec.default_guidance_scale is not None
            else spec.serving_noop_guidance_scale
        )
        request = dict(
            request_id="startup-smoke",
            model=self.model_id,
            prompt=spec.smoke_prompt,
            height=profile.height,
            width=profile.width,
            num_inference_steps=spec.smoke_steps,
            guidance_scale=guidance,
            seed=0,
        )
        if spec.output != "video":
            return DiffletGenerateRequest(**request)
        from difflet.serving.models._common import StartupSmokeTarget

        if self._smoke is not None:
            self._smoke.cleanup()
        self._smoke = StartupSmokeTarget()
        return DiffletGenerateRequest(
            **request,
            output_format="mp4",
            video=VideoGenerateOptions(
                num_frames=int(profile.num_frames),
                fps=int(profile.output_fps),
                output_target=self._smoke.file_target(),
            ),
        )

    def validate_smoke_output(self, output) -> None:
        spec = self.spec
        profile = self.active_profile
        if spec.output == "video":
            from difflet.serving.models._common import validate_smoke_output

            if profile is None or self._smoke is None:
                raise RuntimeError(f"{spec.display_name} smoke ran without a profile or a target")
            try:
                validate_smoke_output(output, profile=profile, expected_path=self._smoke.path)
            finally:
                self._smoke.cleanup()
                self._smoke = None
        else:
            if not output.data:
                raise RuntimeError(f"{spec.display_name} smoke produced empty output")
            from PIL import Image

            size = Image.open(io.BytesIO(output.data)).size
            if profile is None or size != (profile.width, profile.height):
                raise RuntimeError(
                    f"{spec.display_name} smoke image is {size}, not the profile shape"
                )
        print(f"[difflet serve] {spec.display_name} generation smoke passed")

    def reset_request_state(self, outcome: str) -> None:
        return None

    async def shutdown(self) -> None:
        if self._smoke is not None:
            self._smoke.cleanup()
            self._smoke = None
        self.pipe = None
        self.active_profile = None
        self.active_runtime = None


__all__ = [
    "STAGE_ID",
    "GpuImagePayload",
    "GpuInitialPayload",
    "GpuPipelineRunner",
    "GpuServingArtifactPreparer",
    "GpuServingRequestValidator",
    "GpuServingStageAdapter",
    "GpuVideoPayload",
    "gpu_serving_metadata",
    "refuse_unsupported_profile",
    "resident_runtime_plan",
]
