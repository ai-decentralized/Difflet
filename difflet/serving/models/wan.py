"""Wan 2.1 resident video serving adapter with host VAE decode."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from difflet.pipeline.compile_cache import CacheSpec
from difflet.registry import resolve_model
from difflet.serving.artifact_manager import ImmutableArtifactManager
from difflet.serving.engines.stage_pipeline import (
    ErasedStageRunner,
    ValidatedStageRunner,
    require_exact_payload,
    stage_result,
)
from difflet.serving.errors import invalid_extra_body, prompt_too_long, profile_mismatch
from difflet.serving.models._common import (
    StartupSmokeTarget,
    compiled_model_payloads_ready,
    encode_video_tensor,
    require_video_target,
    resident_environment,
    serving_compile_environment,
    validate_smoke_output,
)
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators.base import resolve_hf_model_source, validate_guidance_scale
from difflet.serving.types import (
    ArtifactPublishTarget,
    ArtifactSet,
    CompileArtifactIdentity,
    DiffletCompileSpec,
    DiffletGenerateRequest,
    FileBackedGenerateOutput,
    ParallelTopology,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    RuntimePlan,
    ServingProfile,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
    StageRuntimeSpec,
)

_MODEL_TYPE = "wan"
_HF_MODEL_IDS = (
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
)
_HF_MODEL_ID = _HF_MODEL_IDS[0]
_TEXT_SEQ_LEN = 512
_MAX_GUIDANCE_SCALE = 20.0
_FPS = 16
_TP_DEGREE = 4
_WORLD_SIZE = 4
_VAE_TEMPORAL_SCALE = 4
_VAE_SPATIAL_SCALE = 8
_PATCH_SIZE = 2

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WanInitialPayload(StagePayload):
    pass


@dataclass(frozen=True, slots=True)
class WanPromptPayload(StagePayload):
    prompt_embeds: Any
    negative_prompt_embeds: Any | None


@dataclass(frozen=True, slots=True)
class WanLatentPayload(StagePayload):
    latents: Any


@dataclass(frozen=True, slots=True)
class WanFinalPayload(StagePayload):
    output: FileBackedGenerateOutput


class WanServingArtifactPreparer:
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
        _validate_profile(profile)
        entry = resolve_model(self.model_id, model_type=_MODEL_TYPE)
        source = resolve_hf_model_source(
            self.model_id,
            revision=self.revision,
            download_policy=download_policy,
            allow_patterns=entry.download_patterns,
        )
        spec = _compile_spec(source, profile)
        manager = ImmutableArtifactManager(profile.cache_dir or Path.home() / ".cache" / "difflet")
        binding = manager.prepare(
            model_type=_MODEL_TYPE,
            artifact_id=spec.artifact_id,
            identity=spec.identity,
            policy=compile_policy,
            compile_artifact=lambda target: _compile_artifact(source, profile, spec, target),
            validate_payload=lambda path: _validate_artifact(source, profile, spec, path),
        )
        pipeline = _pipeline_definition()
        runtime_plan = _runtime_plan(profile, pipeline, spec)
        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=pipeline,
            runtime_plan=runtime_plan,
            compile_specs=(spec,),
            artifacts=ArtifactSet((binding,)),
        )


class WanServingRequestValidator:
    def __init__(self, runtime: ResolvedRuntimeBundle) -> None:
        _validate_profile(runtime.profile)
        self.runtime = runtime
        self._tokenizer = None

    def preload(self) -> None:
        self._tokenizer_for_runtime()

    def validate(self, request: DiffletGenerateRequest) -> None:
        profile = self.runtime.profile
        if request.model != profile.model_id:
            raise profile_mismatch("request model does not match the loaded Wan profile")
        if request.output_format != "mp4":
            raise invalid_extra_body("Wan serving only supports MP4 output")
        if (
            request.height != profile.height
            or request.width != profile.width
            or request.video is None
            or request.video.num_frames != profile.num_frames
            or request.video.fps != profile.output_fps
        ):
            raise profile_mismatch("request video shape does not match Wan serving profile")
        validate_guidance_scale(request, maximum=_MAX_GUIDANCE_SCALE)
        if request.video.guidance_scale_2 is not None:
            raise invalid_extra_body(
                "guidance_scale_2 is only available for a verified Wan 2.2 adapter"
            )
        if request.video.boundary_ratio is not None:
            raise invalid_extra_body(
                "boundary_ratio is only available for a verified Wan 2.2 adapter"
            )
        if request.video.flow_shift is not None:
            raise invalid_extra_body("flow_shift is not request-safe in Wan P0 serving")
        if request.video.true_cfg_scale is not None:
            raise invalid_extra_body("true_cfg_scale is not supported by Wan P0 serving")
        for name, text in (
            ("prompt", request.prompt),
            ("negative_prompt", request.video.negative_prompt),
        ):
            if text is None:
                continue
            encoded = self._tokenizer_for_runtime()(
                text,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )
            if int(encoded.input_ids.shape[1]) > _TEXT_SEQ_LEN:
                raise prompt_too_long(f"Wan {name} exceeds text bucket {_TEXT_SEQ_LEN}")

    def _tokenizer_for_runtime(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(Path(self.runtime.source.pinned_model_path) / "tokenizer")
            )
        return self._tokenizer


class WanPromptEncoderStageRunner:
    def __init__(self, adapter: "WanServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[WanInitialPayload],
    ) -> StageExecutionResult[WanPromptPayload]:
        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        pipeline = self.adapter._require_application().pipeline
        prompt_embeds = pipeline.encode_prompt(prompt=invocation.request.prompt)
        if prompt_embeds is None:
            raise RuntimeError("Wan prompt encoder returned no embeddings")
        video = invocation.request.video
        assert video is not None
        negative_prompt_embeds = None
        if invocation.request.guidance_scale > 1.0 or video.negative_prompt is not None:
            negative_prompt_embeds = pipeline.encode_prompt(prompt=video.negative_prompt or "")
        invocation.context.cancellation.throw_if_cancelled()
        return stage_result(
            WanPromptPayload(prompt_embeds, negative_prompt_embeds),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        return None


class WanDenoiserStageRunner:
    def __init__(self, adapter: "WanServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[WanPromptPayload],
    ) -> StageExecutionResult[WanLatentPayload]:
        import torch

        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        request = invocation.request
        assert request.video is not None
        output = self.adapter._require_application().pipeline(
            prompt_embeds=invocation.input.prompt_embeds,
            negative_prompt_embeds=invocation.input.negative_prompt_embeds,
            generator=torch.Generator().manual_seed(request.seed),
            height=request.height,
            width=request.width,
            num_frames=request.video.num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            output_type="latent",
        )
        invocation.context.cancellation.throw_if_cancelled()
        return stage_result(
            WanLatentPayload(output.latents),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        return None


class WanHostDecoderStageRunner:
    def __init__(self, adapter: "WanServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[WanLatentPayload],
    ) -> StageExecutionResult[WanFinalPayload]:
        import torch

        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        vae = self.adapter._require_vae()
        latents = invocation.input.latents.detach().to(device="cpu", dtype=torch.float32)
        mean = torch.tensor(vae.config.latents_mean).view(1, int(vae.config.z_dim), 1, 1, 1)
        inverse_std = 1.0 / torch.tensor(vae.config.latents_std).view(
            1, int(vae.config.z_dim), 1, 1, 1
        )
        latents = latents / inverse_std + mean
        with torch.no_grad():
            frames = vae.decode(latents, return_dict=False)[0]
        frames = frames.to(torch.float32).clamp(-1.0, 1.0)
        invocation.context.cancellation.throw_if_cancelled()
        output = encode_video_tensor(
            frames,
            invocation.request,
            layout="BCTHW",
            value_range="minus_one_to_one",
        )
        invocation.context.cancellation.throw_if_cancelled()
        return stage_result(WanFinalPayload(output), started_monotonic=started)

    async def shutdown(self) -> None:
        return None


class WanServingStageAdapter:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.runtime: ResolvedRuntimeBundle | None = None
        self.profile: ServingProfile | None = None
        self.application = None
        self.vae = None
        self._smoke: StartupSmokeTarget | None = None

    async def create_loaded_runners(
        self,
        runtime: ResolvedRuntimeBundle,
    ) -> OrderedDict[str, ErasedStageRunner]:
        _validate_profile(runtime.profile)
        binding = runtime.artifacts.require("generation")
        spec = runtime.require_compile_spec("generation")
        manager = ImmutableArtifactManager(
            runtime.profile.cache_dir or Path.home() / ".cache" / "difflet"
        )
        manager.validate_binding(
            binding,
            validate_payload=lambda path: _validate_artifact(
                runtime.source, runtime.profile, spec, path
            ),
        )
        application = _build_application(runtime.source, runtime.profile)
        application.load(
            str(binding.path),
            start_rank_id=0,
            local_ranks_size=runtime.profile.world_size,
            skip_warmup=True,
        )
        self.application = application
        self.vae = _load_host_vae(runtime.source.pinned_model_path)
        self.runtime = runtime
        self.profile = runtime.profile
        return OrderedDict(
            (
                (
                    "prompt_encoder",
                    ValidatedStageRunner(
                        WanPromptEncoderStageRunner(self),
                        WanInitialPayload,
                        WanPromptPayload,
                    ),
                ),
                (
                    "denoiser",
                    ValidatedStageRunner(
                        WanDenoiserStageRunner(self),
                        WanPromptPayload,
                        WanLatentPayload,
                    ),
                ),
                (
                    "decoder",
                    ValidatedStageRunner(
                        WanHostDecoderStageRunner(self),
                        WanLatentPayload,
                        WanFinalPayload,
                    ),
                ),
            )
        )

    def initial_payload(self, request: DiffletGenerateRequest) -> WanInitialPayload:
        require_video_target(request)
        return WanInitialPayload()

    def finalize(self, payload: StagePayload) -> FileBackedGenerateOutput:
        return require_exact_payload(payload, WanFinalPayload, boundary="Wan final payload").output

    def smoke_request(self) -> DiffletGenerateRequest:
        if self.profile is None:
            raise RuntimeError("Wan serving profile is not loaded")
        if self._smoke is not None:
            self._smoke.cleanup()
        self._smoke = StartupSmokeTarget()
        profile = self.profile
        assert profile.num_frames is not None and profile.output_fps is not None
        from difflet.serving.types import VideoGenerateOptions

        return DiffletGenerateRequest(
            request_id="startup-smoke",
            model=self.model_id,
            prompt="a small red square",
            height=profile.height,
            width=profile.width,
            num_inference_steps=1,
            guidance_scale=1.0,
            seed=0,
            output_format="mp4",
            video=VideoGenerateOptions(
                num_frames=profile.num_frames,
                fps=profile.output_fps,
                output_target=self._smoke.file_target(),
            ),
        )

    def validate_smoke_output(self, output: FileBackedGenerateOutput) -> None:
        if self.profile is None or self._smoke is None:
            raise RuntimeError("Wan serving profile is not loaded")
        try:
            validate_smoke_output(
                output,
                profile=self.profile,
                expected_path=self._smoke.path,
            )
        finally:
            if self._smoke is not None:
                self._smoke.cleanup()
                self._smoke = None

    def reset_request_state(self, outcome: str) -> None:
        if outcome == "error" and self._smoke is not None:
            self._smoke.cleanup()
            self._smoke = None

    async def shutdown(self) -> None:
        if self._smoke is not None:
            self._smoke.cleanup()
        self._smoke = None
        self.vae = None
        self.application = None
        self.profile = None
        self.runtime = None

    def _require_application(self):
        if self.application is None:
            raise RuntimeError("Wan application is not loaded")
        return self.application

    def _require_vae(self):
        if self.vae is None:
            raise RuntimeError("Wan host VAE is not loaded")
        return self.vae


def _validate_profile(profile: ServingProfile) -> None:
    if profile.model_type != _MODEL_TYPE or profile.model_id not in _HF_MODEL_IDS:
        raise ValueError("Wan serving adapter only supports the enabled Wan checkpoints")
    if profile.output_modality != "video" or profile.output_mime_type != "video/mp4":
        raise ValueError("Wan serving profile must produce video/mp4")
    if profile.dtype.lower().removeprefix("torch.") not in {"bf16", "bfloat16"}:
        raise ValueError("Wan resident serving requires bfloat16")
    spatial_multiple = _VAE_SPATIAL_SCALE * _PATCH_SIZE
    for value, name in ((profile.height, "height"), (profile.width, "width")):
        if type(value) is not int or value <= 0 or value % spatial_multiple:
            raise ValueError(
                f"Wan serving {name} must be a positive integer divisible by {spatial_multiple}"
            )
    if type(profile.num_frames) is not int or profile.num_frames <= 0:
        raise ValueError("Wan serving num_frames must be a positive integer")
    if (profile.num_frames - 1) % _VAE_TEMPORAL_SCALE:
        raise ValueError(
            "Wan serving num_frames must equal 4n+1 for exact causal VAE reconstruction"
        )
    if not profile.host_vae:
        raise ValueError("Wan resident serving requires host VAE decode")
    if profile.output_fps != _FPS:
        raise ValueError(f"Wan video profile requires {_FPS} FPS")
    parallel = profile.parallel
    if parallel.tp_degree != _TP_DEGREE:
        raise ValueError("Wan resident serving requires tp_degree=4")
    if parallel.cp_degree != 1:
        raise ValueError("Wan resident serving requires cp_degree=1")
    if parallel.dp_degree != 1:
        raise ValueError("Wan resident serving requires dp_degree=1")
    if parallel.cfg_parallel_enabled:
        raise ValueError("Wan resident serving requires CFG parallel off")
    if profile.world_size != _WORLD_SIZE:
        raise ValueError("Wan resident serving requires world_size=4")
    if profile.teacache_speedup is not None or profile.teacache_calibration_data is not None:
        raise ValueError("Wan resident video serving does not support TeaCache")


def _compile_spec(
    source: ResolvedModelSource,
    profile: ServingProfile,
) -> DiffletCompileSpec:
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
        application_kwargs=_application_kwargs(),
    )
    identity = CompileArtifactIdentity.from_cache_inputs(
        {"component_id": "generation", "cache_inputs": cache_spec.cache_inputs()}
    )
    return DiffletCompileSpec(
        artifact_id="generation",
        component_id="generation",
        identity=identity,
    )


def _application_kwargs() -> dict[str, Any]:
    return {
        "text_seq_len": _TEXT_SEQ_LEN,
        "batch_size": 1,
        "enable_text_encoder": True,
        "enable_transformer": True,
        "enable_transformer_2": False,
        "enable_vae_decoder": False,
    }


def _build_application(source: ResolvedModelSource, profile: ServingProfile):
    from difflet.models.wan.application import NeuronWanApplication

    return NeuronWanApplication(
        model_path=source.pinned_model_path,
        parallel=profile.parallel,
        dtype=_torch_bfloat16(),
        shape=profile.shape_dict(),
        **_application_kwargs(),
    )


def _compile_artifact(
    source: ResolvedModelSource,
    profile: ServingProfile,
    spec: DiffletCompileSpec,
    target: ArtifactPublishTarget,
) -> None:
    _validate_compile_target(spec, target)
    application = _build_application(source, profile)
    with serving_compile_environment(profile.world_size):
        application.compile(str(target.staging_path))
    if not application.has_compiled_artifacts(
        str(target.staging_path)
    ) or not compiled_model_payloads_ready(application, target.staging_path):
        raise RuntimeError("Wan serving compile produced incomplete artifacts")


def _validate_artifact(
    source: ResolvedModelSource,
    profile: ServingProfile,
    spec: DiffletCompileSpec,
    path: Path,
) -> None:
    if spec.component_id != "generation":
        raise ValueError(f"unknown Wan serving component {spec.component_id!r}")
    application = _build_application(source, profile)
    if not application.has_compiled_artifacts(str(path)) or not compiled_model_payloads_ready(
        application, path
    ):
        raise ValueError(f"Wan serving artifact is incomplete at {path}")


def _validate_compile_target(
    spec: DiffletCompileSpec,
    target: ArtifactPublishTarget,
) -> None:
    if target.artifact_id != spec.artifact_id or target.identity != spec.identity:
        raise ValueError("Wan compile target does not match compile spec")


def _runtime_plan(profile, pipeline, spec: DiffletCompileSpec) -> RuntimePlan:
    environment, allocation = resident_environment(profile, allocation_id="wan-resident")
    topology = ParallelTopology(
        tp_degree=profile.parallel.tp_degree,
        cp_degree=profile.parallel.cp_degree,
        world_size=profile.world_size,
    )
    return RuntimePlan(
        mode="resident",
        profile_identity=spec.identity.digest,
        environment=environment,
        allocations=(allocation,),
        stages=(
            StageRuntimeSpec(
                stage_id="prompt_encoder",
                allocation_id=allocation.allocation_id,
                topology=topology,
                artifact_id=spec.artifact_id,
            ),
            StageRuntimeSpec(
                stage_id="denoiser",
                allocation_id=allocation.allocation_id,
                topology=topology,
                artifact_id=spec.artifact_id,
            ),
            StageRuntimeSpec(
                stage_id="decoder",
                allocation_id=None,
                topology=None,
                artifact_id=None,
                placement="host",
            ),
        ),
    )


def _pipeline_definition():
    from difflet.common.registry.wan import serving_metadata

    return serving_metadata().pipeline_definition


def _load_host_vae(model_path: str):
    import torch
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        str(Path(model_path) / "vae"), torch_dtype=torch.float32
    ).eval()


def _torch_bfloat16():
    import torch

    return torch.bfloat16


__all__ = [
    "WanDenoiserStageRunner",
    "WanHostDecoderStageRunner",
    "WanPromptEncoderStageRunner",
    "WanServingArtifactPreparer",
    "WanServingRequestValidator",
    "WanServingStageAdapter",
]
