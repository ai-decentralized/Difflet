"""Resident LTX-2 video serving on the existing hybrid CPU/Neuron pipeline."""

from __future__ import annotations

import math
import os
import stat
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from difflet.common.neuron_cores import (
    resolve_available_neuron_core_ids,
    select_neuron_core_ids,
)
from difflet.pipeline.compile_cache import CacheSpec
from difflet.registry import resolve_model
from difflet.serving.artifact_manager import ImmutableArtifactManager
from difflet.serving.engines.stage_pipeline import (
    ErasedStageRunner,
    ValidatedStageRunner,
    require_exact_payload,
    stage_result,
)
from difflet.serving.errors import invalid_extra_body, profile_mismatch, prompt_too_long
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators.base import resolve_hf_model_source
from difflet.serving.types import (
    ArtifactPublishTarget,
    ArtifactSet,
    CompileArtifactIdentity,
    DiffletCompileSpec,
    DiffletGenerateRequest,
    DistributedProcessEnvironment,
    FileBackedGenerateOutput,
    FileOutputTarget,
    GenerateOutput,
    ParallelTopology,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    RuntimeEnvironment,
    RuntimePlan,
    ServingProfile,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
    StageRuntimeSpec,
    VideoGenerateOptions,
    WorkerAllocationSpec,
)
from difflet.serving.video_media import (
    VideoMediaMetadata,
    encode_tensor_to_mp4,
    validate_mp4,
)
from difflet.serving.models._common import compiled_model_payloads_ready

_HF_MODEL_ID = "Lightricks/LTX-2"
_MODEL_TYPE = "ltx_2"
_PIPELINE_ARTIFACT_ID = "pipeline"
_FPS = 24
_TP_DEGREE = 4
_WORLD_SIZE = 4
_MAX_INFERENCE_STEPS = 200
_MAX_GUIDANCE_SCALE = 20.0
_MAX_SEED = 2**63 - 1
_TEXT_SEQ_LEN = 1024
_VAE_SPATIAL_SCALE = 32
_VAE_TEMPORAL_SCALE = 8


@dataclass(frozen=True, slots=True)
class LTX2InitialPayload(StagePayload):
    """Nominal input for the one opaque LTX-2 serving stage."""


@dataclass(frozen=True, slots=True)
class LTX2FinalPayload(StagePayload):
    """File-backed MP4 produced by the opaque LTX-2 serving stage."""

    output: FileBackedGenerateOutput


def build_compile_plan(
    source: ResolvedModelSource,
    profile: ServingProfile,
) -> tuple[DiffletCompileSpec, ...]:
    """Build the single immutable artifact identity for one fixed profile."""

    _validate_profile(profile)
    if source.model_id != profile.model_id:
        raise ValueError("LTX-2 source and serving profile model IDs must match")
    cache_spec = CacheSpec(
        model_id=source.model_id,
        model_path=source.pinned_model_path,
        model_name=_MODEL_TYPE,
        parallel=profile.parallel,
        dtype=_torch_bfloat16(),
        height=profile.height,
        width=profile.width,
        num_frames=profile.num_frames,
        revision=source.resolved_source_id,
        application_kwargs=None,
    )
    identity = CompileArtifactIdentity.from_cache_inputs(
        {
            "component_id": _PIPELINE_ARTIFACT_ID,
            "cache_inputs": cache_spec.cache_inputs(),
        }
    )
    return (
        DiffletCompileSpec(
            artifact_id=_PIPELINE_ARTIFACT_ID,
            component_id=_PIPELINE_ARTIFACT_ID,
            identity=identity,
        ),
    )


def build_pipeline(
    source: ResolvedModelSource,
    profile: ServingProfile,
    *,
    load: bool,
    force_compile: bool = False,
    skip_compile: bool = False,
    compiled_path_override: str | Path,
    host_components: bool,
):
    """Construct the existing LTX-2 lower pipeline in commit-bound mode."""

    _validate_profile(profile)
    if source.model_id != profile.model_id:
        raise ValueError("LTX-2 source and serving profile model IDs must match")
    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    application_kwargs = None
    if host_components:
        application_kwargs = {
            "enable_host_pipeline": True,
            "enable_decode_components": True,
            "host_device": "cpu",
        }
    return DiffletPipeline.from_pretrained(
        source.model_id,
        model_type=_MODEL_TYPE,
        parallel=profile.parallel,
        dtype=_torch_bfloat16(),
        height=profile.height,
        width=profile.width,
        num_frames=profile.num_frames,
        compile_cache_dir=profile.cache_dir,
        revision=profile.revision,
        local_files_only=True,
        force_compile=force_compile,
        skip_compile=skip_compile,
        load=load,
        skip_warmup=load,
        application_kwargs=application_kwargs,
        model_path_override=source.pinned_model_path,
        resolved_source_id=source.resolved_source_id,
        compiled_path_override=str(compiled_path_override),
    )


def pipeline_artifacts_ready(pipe: Any) -> bool:
    """Return whether the fixed-profile compile payload is complete."""

    from difflet.pipeline.compile_cache import has_valid_manifest

    if not has_valid_manifest(pipe.compiled_path, pipe.cache_spec):
        return False
    checker = getattr(pipe.app, "has_compiled_artifacts", None)
    if checker is None or not bool(checker(str(pipe.compiled_path))):
        return False
    return compiled_model_payloads_ready(pipe.app, pipe.compiled_path)


def compile_serving_artifact(
    source: ResolvedModelSource,
    profile: ServingProfile,
    spec: DiffletCompileSpec,
    target: ArtifactPublishTarget,
) -> None:
    """Compile LTX-2 into the artifact manager's private staging directory."""

    _validate_spec_target(spec, target)
    _validate_profile(profile)
    with _serving_compile_environment():
        pipe = build_pipeline(
            source,
            profile,
            load=False,
            force_compile=True,
            compiled_path_override=target.staging_path,
            host_components=False,
        )
    if not pipeline_artifacts_ready(pipe):
        raise ValueError(f"LTX-2 compile produced incomplete payload at {target.staging_path}")


def validate_compiled_artifact(
    source: ResolvedModelSource,
    profile: ServingProfile,
    spec: DiffletCompileSpec,
    artifact_root: Path,
) -> None:
    """Validate a published LTX-2 artifact without loading model weights."""

    if spec.component_id != _PIPELINE_ARTIFACT_ID:
        raise ValueError(f"unknown LTX-2 component {spec.component_id!r}")
    pipe = build_pipeline(
        source,
        profile,
        load=False,
        skip_compile=True,
        compiled_path_override=artifact_root,
        host_components=False,
    )
    if not pipeline_artifacts_ready(pipe):
        raise ValueError(f"LTX-2 artifact payload is incomplete at {artifact_root}")


def build_runtime_plan(
    profile: ServingProfile,
    pipeline_definition: Any,
    specs: tuple[DiffletCompileSpec, ...],
) -> RuntimePlan:
    """Describe one TP4 hybrid resident worker with host encode/decode."""

    _validate_profile(profile)
    if len(specs) != 1:
        raise ValueError("LTX-2 serving requires exactly one compile spec")
    spec = specs[0]
    if spec.artifact_id != _PIPELINE_ARTIFACT_ID or spec.component_id != _PIPELINE_ARTIFACT_ID:
        raise ValueError("LTX-2 serving requires the pipeline compile artifact")
    if len(pipeline_definition.stages) != 1:
        raise ValueError("LTX-2 serving requires exactly one opaque pipeline stage")
    stage_definition = pipeline_definition.stages[0]
    if stage_definition.stage_id != _PIPELINE_ARTIFACT_ID:
        raise ValueError("LTX-2 serving stage must be named 'pipeline'")

    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment(
        available_core_ids=resolve_available_neuron_core_ids(required_num_cores=_WORLD_SIZE),
        num_cores_override=None,
        virtual_core_size_override=None,
        logical_nc_config_override=None,
        inherited_distributed=distributed,
        child_distributed=distributed,
    )
    allocation = WorkerAllocationSpec(
        allocation_id="ltx-2-resident",
        requested_num_cores=_WORLD_SIZE,
        effective_num_cores=_WORLD_SIZE,
        world_size=_WORLD_SIZE,
    )
    return RuntimePlan(
        mode="resident",
        profile_identity=spec.identity.digest,
        environment=environment,
        allocations=(allocation,),
        stages=(
            StageRuntimeSpec(
                stage_id=stage_definition.stage_id,
                allocation_id=allocation.allocation_id,
                topology=ParallelTopology(
                    tp_degree=_TP_DEGREE,
                    cp_degree=1,
                    world_size=_WORLD_SIZE,
                ),
                artifact_id=spec.artifact_id,
                placement="hybrid",
            ),
        ),
    )


class LTX2ServingArtifactPreparer:
    """Resolve commit-pinned weights and publish one immutable TP4 artifact."""

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
        _validate_profile(profile, expected_model_id=self.model_id)
        if profile.revision != self.revision:
            raise ValueError("LTX-2 preparer revision must match the serving profile")
        entry = resolve_model(self.model_id, model_type=_MODEL_TYPE)
        print(f"[difflet serve] resolving LTX-2 weights for {self.model_id}")
        source = resolve_hf_model_source(
            self.model_id,
            revision=self.revision,
            download_policy=download_policy,
            allow_patterns=entry.download_patterns,
        )
        specs = build_compile_plan(source, profile)
        manager = ImmutableArtifactManager(profile.cache_dir or Path.home() / ".cache" / "difflet")

        def _prepare_binding(spec: DiffletCompileSpec):
            return manager.prepare(
                model_type=self.model_type,
                artifact_id=spec.artifact_id,
                identity=spec.identity,
                policy=compile_policy,
                compile_artifact=lambda target: compile_serving_artifact(
                    source, profile, spec, target
                ),
                validate_payload=lambda path: validate_compiled_artifact(
                    source, profile, spec, path
                ),
            )

        bindings = tuple(_prepare_binding(spec) for spec in specs)
        pipeline_definition = _pipeline_definition()
        runtime_plan = build_runtime_plan(profile, pipeline_definition, specs)
        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=pipeline_definition,
            runtime_plan=runtime_plan,
            compile_specs=specs,
            artifacts=ArtifactSet(bindings),
        )


class LTX2ServingRequestValidator:
    """Reject requests that cannot use the process's fixed LTX-2 profile."""

    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, runtime: ResolvedRuntimeBundle) -> None:
        _validate_profile(runtime.profile)
        self.runtime = runtime
        self.model_id = runtime.profile.model_id
        self._tokenizer = None

    def preload(self) -> None:
        self._tokenizer_for_runtime()

    def validate(self, request: DiffletGenerateRequest) -> None:
        profile = self.runtime.profile
        if request.model != profile.model_id:
            raise profile_mismatch("request model does not match the loaded LTX-2 profile")
        if (request.height, request.width) != (profile.height, profile.width):
            raise profile_mismatch("request width and height do not match the loaded LTX-2 profile")
        if request.output_format != "mp4":
            raise invalid_extra_body("LTX-2 serving only supports MP4 output")
        if type(request.num_inference_steps) is not int or not (
            2 <= request.num_inference_steps <= _MAX_INFERENCE_STEPS
        ):
            raise invalid_extra_body(
                "LTX-2 num_inference_steps must be an integer between 2 and 200"
            )
        if type(request.guidance_scale) not in (int, float) or not math.isfinite(
            float(request.guidance_scale)
        ):
            raise invalid_extra_body("LTX-2 guidance_scale must be a finite number")
        if not 0.0 <= float(request.guidance_scale) <= _MAX_GUIDANCE_SCALE:
            raise invalid_extra_body("LTX-2 guidance_scale must be between 0 and 20")
        if type(request.seed) is not int or not 0 <= request.seed <= _MAX_SEED:
            raise invalid_extra_body(f"LTX-2 seed must be an integer between 0 and {_MAX_SEED}")
        video = request.video
        if video is None:
            raise invalid_extra_body("LTX-2 serving requires video request options")
        if video.num_frames != profile.num_frames:
            raise profile_mismatch("request num_frames does not match the loaded LTX-2 profile")
        if video.fps != profile.output_fps:
            raise profile_mismatch("request fps does not match the loaded LTX-2 profile")
        for field_name in (
            "guidance_scale_2",
            "boundary_ratio",
            "flow_shift",
            "true_cfg_scale",
        ):
            if getattr(video, field_name) is not None:
                raise invalid_extra_body(f"{field_name} is not supported by LTX-2 serving")
        for name, text in (("prompt", request.prompt), ("negative_prompt", video.negative_prompt)):
            if text is None:
                continue
            encoded = self._tokenizer_for_runtime()(
                text,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )
            if int(encoded.input_ids.shape[1]) > _TEXT_SEQ_LEN:
                raise prompt_too_long(f"LTX-2 {name} exceeds text bucket {_TEXT_SEQ_LEN}")
        if video.output_target is not None:
            try:
                _validate_output_target(video.output_target, require_existing=False)
            except (OSError, TypeError, ValueError) as exc:
                raise invalid_extra_body(f"invalid LTX-2 output target: {exc}") from exc

    def _tokenizer_for_runtime(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(Path(self.runtime.source.pinned_model_path) / "tokenizer")
            )
        return self._tokenizer


class LTX2PipelineRunner:
    """Run the existing hybrid pipeline and encode its frames to the parent target."""

    def __init__(self, pipe: Any, profile: ServingProfile) -> None:
        _validate_profile(profile)
        self.pipe = pipe
        self.profile = profile

    async def execute(
        self,
        invocation: StageInvocation[LTX2InitialPayload],
    ) -> StageExecutionResult[LTX2FinalPayload]:
        started = time.monotonic()
        request = invocation.request
        context = invocation.context
        context.cancellation.throw_if_cancelled()
        video, target = _require_video_target(request)
        output = self.pipe(
            prompt=request.prompt,
            negative_prompt=video.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=_seeded_generator(request.seed),
            output_type="pt",
        )
        context.cancellation.throw_if_cancelled()
        frames = output.frames if hasattr(output, "frames") else output[0]
        frames = frames.float().clamp(0.0, 1.0)
        media = encode_tensor_to_mp4(
            frames,
            target.staging_path,
            fps=video.fps,
            layout="BFCHW",
            value_range="zero_to_one",
        )
        context.cancellation.throw_if_cancelled()
        _validate_generated_media(media, self.profile)
        return stage_result(
            LTX2FinalPayload(
                FileBackedGenerateOutput(
                    path=target.staging_path,
                    mime_type=target.mime_type,
                    output_format=target.output_format,
                    size_bytes=media.size_bytes,
                    width=media.width,
                    height=media.height,
                    num_frames=media.num_frames,
                    fps=media.fps,
                    duration_s=media.duration_s,
                )
            ),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        self.pipe = None


class LTX2ServingStageAdapter:
    """Bridge one opaque LTX-2 pipeline into the generic resident stage engine."""

    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.active_runtime: ResolvedRuntimeBundle | None = None
        self.active_profile: ServingProfile | None = None
        self._untransferred_pipe: Any | None = None
        self._smoke_directory: tempfile.TemporaryDirectory[str] | None = None
        self._smoke_target: Path | None = None

    async def create_loaded_runners(
        self,
        runtime: ResolvedRuntimeBundle,
    ) -> OrderedDict[str, ErasedStageRunner]:
        profile = runtime.profile
        _validate_profile(profile, expected_model_id=self.model_id)
        print(f"[difflet serve] loading LTX-2 worker profile {profile}")
        self.active_runtime = runtime
        self.active_profile = profile
        binding = runtime.artifacts.require(_PIPELINE_ARTIFACT_ID)
        spec = runtime.require_compile_spec(_PIPELINE_ARTIFACT_ID)
        manager = ImmutableArtifactManager(profile.cache_dir or Path.home() / ".cache" / "difflet")
        manager.validate_binding(
            binding,
            validate_payload=lambda path: validate_compiled_artifact(
                runtime.source, profile, spec, path
            ),
        )
        pipe = build_pipeline(
            runtime.source,
            profile,
            load=True,
            skip_compile=True,
            compiled_path_override=binding.path,
            host_components=True,
        )
        self._untransferred_pipe = pipe
        runner = ValidatedStageRunner(
            LTX2PipelineRunner(pipe, profile),
            LTX2InitialPayload,
            LTX2FinalPayload,
        )
        self._untransferred_pipe = None
        print("[difflet serve] LTX-2 worker loaded")
        return OrderedDict((("pipeline", runner),))

    def initial_payload(self, request: DiffletGenerateRequest) -> LTX2InitialPayload:
        return LTX2InitialPayload()

    def finalize(self, payload: StagePayload) -> FileBackedGenerateOutput:
        return require_exact_payload(
            payload,
            LTX2FinalPayload,
            boundary="LTX-2 final payload",
        ).output

    def smoke_request(self) -> DiffletGenerateRequest:
        if self.active_profile is None:
            raise RuntimeError("LTX-2 serving profile is not loaded")
        self._cleanup_smoke_target()
        profile = self.active_profile
        directory = tempfile.TemporaryDirectory(prefix="difflet-ltx2-smoke-")
        target = Path(directory.name).resolve() / "startup-smoke.part.mp4"
        target.touch(mode=0o600, exist_ok=False)
        self._smoke_directory = directory
        self._smoke_target = target
        print("[difflet serve] running LTX-2 generation smoke")
        return DiffletGenerateRequest(
            request_id="startup-smoke",
            model=profile.model_id,
            prompt="a small red square moving slowly",
            height=profile.height,
            width=profile.width,
            num_inference_steps=2,
            guidance_scale=1.0,
            seed=0,
            output_format="mp4",
            video=VideoGenerateOptions(
                num_frames=_require_profile_int(profile.num_frames, "num_frames"),
                fps=_require_profile_int(profile.output_fps, "output_fps"),
                output_target=FileOutputTarget(staging_path=str(target)),
            ),
        )

    def validate_smoke_output(self, output: GenerateOutput) -> None:
        profile = self.active_profile
        target = self._smoke_target
        try:
            if profile is None or target is None:
                raise RuntimeError("LTX-2 smoke target is not active")
            if not isinstance(output, FileBackedGenerateOutput):
                raise RuntimeError("LTX-2 smoke did not produce file-backed output")
            if output.path != str(target):
                raise RuntimeError("LTX-2 smoke wrote outside its temporary target")
            _validate_file_output(output, profile)
            validate_mp4(
                target,
                expected_width=profile.width,
                expected_height=profile.height,
                expected_num_frames=profile.num_frames,
                expected_fps=profile.output_fps,
                require_silent=True,
            )
            print("[difflet serve] LTX-2 generation smoke passed")
        finally:
            self._cleanup_smoke_target()

    def reset_request_state(self, outcome: str) -> None:
        if outcome == "error":
            self._cleanup_smoke_target()

    async def shutdown(self) -> None:
        self._cleanup_smoke_target()
        self._untransferred_pipe = None
        self.active_profile = None
        self.active_runtime = None

    def _cleanup_smoke_target(self) -> None:
        directory = self._smoke_directory
        self._smoke_target = None
        self._smoke_directory = None
        if directory is not None:
            directory.cleanup()


def _pipeline_definition():
    from difflet.common.registry.ltx_2 import serving_metadata

    return serving_metadata().pipeline_definition


def _require_video_target(
    request: DiffletGenerateRequest,
) -> tuple[VideoGenerateOptions, FileOutputTarget]:
    if request.output_format != "mp4":
        raise ValueError("LTX-2 worker requires MP4 output")
    video = request.video
    if video is None:
        raise ValueError("LTX-2 worker requires video request options")
    target = video.output_target
    if target is None:
        raise ValueError("LTX-2 worker requires a parent-owned output target")
    _validate_output_target(target, require_existing=True)
    return video, target


def _validate_output_target(
    target: FileOutputTarget,
    *,
    require_existing: bool,
) -> Path:
    if not isinstance(target, FileOutputTarget):
        raise TypeError("output target must be FileOutputTarget")
    if target.mime_type != "video/mp4" or target.output_format != "mp4":
        raise ValueError("output target must use video/mp4 and mp4")
    path = Path(target.staging_path)
    if not path.is_absolute():
        raise ValueError("output target path must be absolute")
    if not path.name.endswith(".part.mp4"):
        raise ValueError("output target path must end with .part.mp4")
    parent_info = path.parent.lstat()
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError("output target parent must be a real directory")
    if os.path.lexists(path):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("output target must be a regular non-symlink file")
    elif require_existing:
        raise ValueError("output target must be pre-created by the parent process")
    return path


def _validate_generated_media(media: VideoMediaMetadata, profile: ServingProfile) -> None:
    expected = (
        profile.width,
        profile.height,
        profile.num_frames,
        float(_require_profile_int(profile.output_fps, "output_fps")),
    )
    actual = (media.width, media.height, media.num_frames, float(media.fps))
    if actual != expected:
        raise RuntimeError(
            "LTX-2 encoded media does not match the serving profile: "
            f"expected {expected!r}, got {actual!r}"
        )


def _validate_file_output(output: FileBackedGenerateOutput, profile: ServingProfile) -> None:
    if output.mime_type != "video/mp4" or output.output_format != "mp4":
        raise RuntimeError("LTX-2 output is not MP4")
    if output.size_bytes <= 0:
        raise RuntimeError("LTX-2 output is empty")
    expected = (
        profile.width,
        profile.height,
        profile.num_frames,
        float(_require_profile_int(profile.output_fps, "output_fps")),
    )
    actual = (output.width, output.height, output.num_frames, float(output.fps))
    if actual != expected:
        raise RuntimeError(
            "LTX-2 output metadata does not match the serving profile: "
            f"expected {expected!r}, got {actual!r}"
        )


def _validate_profile(
    profile: ServingProfile,
    *,
    expected_model_id: str | None = None,
) -> None:
    if profile.model_type != _MODEL_TYPE:
        raise ValueError("LTX-2 serving profile must use model_type='ltx_2'")
    required_model_id = expected_model_id or _HF_MODEL_ID
    if profile.model_id != required_model_id:
        raise ValueError("LTX-2 serving profile model ID does not match the adapter")
    if profile.output_modality != "video" or profile.output_mime_type != "video/mp4":
        raise ValueError("LTX-2 serving profile must produce video/mp4")
    if profile.dtype.lower().removeprefix("torch.") not in {"bf16", "bfloat16"}:
        raise ValueError("LTX-2 serving requires bfloat16")
    for value, name in ((profile.height, "height"), (profile.width, "width")):
        if type(value) is not int or value <= 0 or value % _VAE_SPATIAL_SCALE:
            raise ValueError(
                f"LTX-2 serving {name} must be a positive integer divisible by "
                f"{_VAE_SPATIAL_SCALE}"
            )
    num_frames = _require_profile_int(profile.num_frames, "num_frames")
    if (num_frames - 1) % _VAE_TEMPORAL_SCALE:
        raise ValueError(
            "LTX-2 serving num_frames must equal 8n+1 for exact causal VAE reconstruction"
        )
    if profile.output_fps != _FPS:
        raise ValueError(f"LTX-2 serving requires {_FPS} fps")
    if not profile.host_vae:
        raise ValueError("LTX-2 serving requires host decode components")
    parallel = profile.parallel
    if (
        parallel.tp_degree != _TP_DEGREE
        or parallel.cp_degree != 1
        or parallel.cfg_parallel_enabled
        or parallel.sp_enabled
        or parallel.dp_degree != 1
        or profile.world_size != _WORLD_SIZE
    ):
        raise ValueError("LTX-2 serving requires TP4, CP1, DP1, CFG off, and SP off")
    if (
        profile.teacache_speedup is not None
        or profile.teacache_calibration_data is not None
        or profile.teacache_cadence is not None
        or profile.teacache_online_delta is not None
    ):
        raise ValueError("LTX-2 resident serving does not support TeaCache")


def _require_profile_int(value: int | None, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"LTX-2 serving {field_name} must be a positive integer")
    return value


def _validate_spec_target(
    spec: DiffletCompileSpec,
    target: ArtifactPublishTarget,
) -> None:
    if spec.component_id != _PIPELINE_ARTIFACT_ID:
        raise ValueError(f"unknown LTX-2 component {spec.component_id!r}")
    if spec.artifact_id != target.artifact_id or spec.identity != target.identity:
        raise ValueError("LTX-2 compile target does not match compile spec")


@contextmanager
def _serving_compile_environment() -> Iterator[None]:
    names = (
        "NEURON_RT_VISIBLE_CORES",
        "NEURON_RT_NUM_CORES",
        "NEURON_RT_VIRTUAL_CORE_SIZE",
        "NEURON_LOGICAL_NC_CONFIG",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
    )
    original = {name: os.environ.get(name) for name in names}
    try:
        visible_core_ids = select_neuron_core_ids(required_num_cores=_WORLD_SIZE)
        os.environ["NEURON_RT_VISIBLE_CORES"] = ",".join(
            str(core_id) for core_id in visible_core_ids
        )
        os.environ["NEURON_RT_NUM_CORES"] = str(_WORLD_SIZE)
        os.environ.pop("NEURON_RT_VIRTUAL_CORE_SIZE", None)
        os.environ.pop("NEURON_LOGICAL_NC_CONFIG", None)
        os.environ.update(
            {"WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
        )
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _seeded_generator(seed: int):
    import torch

    return torch.Generator().manual_seed(seed)


def _torch_bfloat16():
    import torch

    return torch.bfloat16


__all__ = [
    "LTX2FinalPayload",
    "LTX2InitialPayload",
    "LTX2PipelineRunner",
    "LTX2ServingArtifactPreparer",
    "LTX2ServingRequestValidator",
    "LTX2ServingStageAdapter",
    "build_compile_plan",
    "build_pipeline",
    "build_runtime_plan",
    "compile_serving_artifact",
    "pipeline_artifacts_ready",
    "validate_compiled_artifact",
]
