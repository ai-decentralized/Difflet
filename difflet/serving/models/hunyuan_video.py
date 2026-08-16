"""HunyuanVideo 1.0 resident adapter with selectable CLIP/VAE placement."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from difflet.pipeline.compile_cache import toolchain_versions
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
    combined_profile_identity,
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
    PipelineDefinition,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    RuntimePlan,
    ServingProfile,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
    StageRuntimeSpec,
)

_MODEL_TYPE = "hunyuan_video"
_HF_MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
_TEXT_SEQ_LEN = 256
_LLAMA_CROP_START = 95
_LLAMA_SEQ_LEN = _TEXT_SEQ_LEN + _LLAMA_CROP_START
_LLAMA_CAPTURE = "layers.29"
_VIRTUAL_CORE_SIZE = 2
_MAX_GUIDANCE_SCALE = 20.0
_FPS = 24
_TP_DEGREE = 4
_WORLD_SIZE = 4
_VAE_TEMPORAL_SCALE = 4
_VAE_SPATIAL_SCALE = 8
_PATCH_SIZE = 2
_LLAMA_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the "
    "following aspects: 1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of "
    "the objects.3. Actions, events, behaviors temporal relationships, physical movement "
    "changes of the objects.4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)


@dataclass(frozen=True, slots=True)
class HunyuanVideoInitialPayload(StagePayload):
    pass


@dataclass(frozen=True, slots=True)
class HunyuanVideoClipPayload(StagePayload):
    pooled_projections: Any


@dataclass(frozen=True, slots=True)
class HunyuanVideoConditioningPayload(StagePayload):
    pooled_projections: Any
    encoder_hidden_states: Any
    encoder_attention_mask: Any


@dataclass(frozen=True, slots=True)
class HunyuanVideoLatentPayload(StagePayload):
    latents: Any


@dataclass(frozen=True, slots=True)
class HunyuanVideoFinalPayload(StagePayload):
    output: FileBackedGenerateOutput


class HunyuanVideoServingArtifactPreparer:
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
        specs = _compile_specs(source, profile)
        manager = ImmutableArtifactManager(profile.cache_dir or Path.home() / ".cache" / "difflet")
        bindings = tuple(
            manager.prepare(
                model_type=_MODEL_TYPE,
                artifact_id=spec.artifact_id,
                identity=spec.identity,
                policy=compile_policy,
                compile_artifact=lambda target, spec=spec: _compile_artifact(
                    source, profile, spec, target
                ),
                validate_payload=lambda path, spec=spec: _validate_artifact(
                    source, profile, spec, path
                ),
            )
            for spec in specs
        )
        pipeline = _pipeline_definition(profile)
        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=pipeline,
            runtime_plan=_runtime_plan(profile, specs),
            compile_specs=specs,
            artifacts=ArtifactSet(bindings),
        )


class HunyuanVideoServingRequestValidator:
    def __init__(self, runtime: ResolvedRuntimeBundle) -> None:
        _validate_profile(runtime.profile)
        self.runtime = runtime
        self._tokenizer = None

    def preload(self) -> None:
        self._tokenizer_for_runtime()

    def validate(self, request: DiffletGenerateRequest) -> None:
        profile = self.runtime.profile
        if request.model != profile.model_id:
            raise profile_mismatch("request model does not match the loaded HunyuanVideo profile")
        if request.output_format != "mp4":
            raise invalid_extra_body("HunyuanVideo serving only supports MP4 output")
        if (
            request.video is None
            or request.height != profile.height
            or request.width != profile.width
            or request.video.num_frames != profile.num_frames
            or request.video.fps != profile.output_fps
        ):
            raise profile_mismatch(
                "request video shape does not match HunyuanVideo serving profile"
            )
        validate_guidance_scale(request, maximum=_MAX_GUIDANCE_SCALE)
        video = request.video
        if video.negative_prompt is not None:
            raise invalid_extra_body("HunyuanVideo 1.0 serving does not use negative_prompt")
        if video.guidance_scale_2 is not None or video.boundary_ratio is not None:
            raise invalid_extra_body(
                "dual-transformer guidance fields are not supported by HunyuanVideo"
            )
        if video.flow_shift is not None:
            raise invalid_extra_body("flow_shift is not request-safe in HunyuanVideo P0 serving")
        if video.true_cfg_scale is not None:
            raise invalid_extra_body("true_cfg_scale is not supported by HunyuanVideo P0 serving")
        encoded = self._tokenizer_for_runtime()(
            _LLAMA_TEMPLATE.format(request.prompt),
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        if int(encoded.input_ids.shape[1]) > _LLAMA_SEQ_LEN:
            raise prompt_too_long(
                f"HunyuanVideo templated prompt exceeds Llama bucket {_LLAMA_SEQ_LEN}"
            )

    def _tokenizer_for_runtime(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(Path(self.runtime.source.pinned_model_path) / "tokenizer")
            )
        return self._tokenizer


class HunyuanVideoHostClipStageRunner:
    def __init__(self, adapter: "HunyuanVideoServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[HunyuanVideoInitialPayload],
    ) -> StageExecutionResult[HunyuanVideoClipPayload]:
        import torch

        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        tokenizer, model = self.adapter._require_host_clip()
        inputs = tokenizer(
            invocation.request.prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        pooled = output.pooler_output.to(torch.bfloat16).cpu().reshape(1, -1)
        invocation.context.cancellation.throw_if_cancelled()
        return stage_result(HunyuanVideoClipPayload(pooled), started_monotonic=started)

    async def shutdown(self) -> None:
        return None


class HunyuanVideoNeuronClipStageRunner:
    def __init__(self, adapter: "HunyuanVideoServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[HunyuanVideoInitialPayload],
    ) -> StageExecutionResult[HunyuanVideoClipPayload]:
        import torch

        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        tokenizer, app = self.adapter._require_neuron_clip()
        input_ids = tokenizer(
            invocation.request.prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(torch.int64)
        output = app(input_ids)
        pooled = output.pooler_output.to(torch.bfloat16).cpu().reshape(1, -1)
        invocation.context.cancellation.throw_if_cancelled()
        return stage_result(HunyuanVideoClipPayload(pooled), started_monotonic=started)

    async def shutdown(self) -> None:
        return None


class HunyuanVideoLlamaStageRunner:
    def __init__(self, adapter: "HunyuanVideoServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[HunyuanVideoClipPayload],
    ) -> StageExecutionResult[HunyuanVideoConditioningPayload]:
        import torch

        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        app, tokenizer = self.adapter._require_llama()
        tokenized = tokenizer(
            _LLAMA_TEMPLATE.format(invocation.request.prompt),
            max_length=_LLAMA_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = tokenized.input_ids.to(torch.int32)
        attention_mask = tokenized.attention_mask.to(torch.int32)
        output = app(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=torch.arange(_LLAMA_SEQ_LEN, dtype=torch.int32).unsqueeze(0),
            sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        )
        hidden = output.captured_tensors[0][:, _LLAMA_CROP_START:].to(torch.bfloat16).cpu()
        mask = attention_mask[:, _LLAMA_CROP_START:].to(torch.int64)
        invocation.context.cancellation.throw_if_cancelled()
        return stage_result(
            HunyuanVideoConditioningPayload(
                pooled_projections=invocation.input.pooled_projections,
                encoder_hidden_states=hidden,
                encoder_attention_mask=mask,
            ),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        return None


class HunyuanVideoDenoiserStageRunner:
    def __init__(self, adapter: "HunyuanVideoServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[HunyuanVideoConditioningPayload],
    ) -> StageExecutionResult[HunyuanVideoLatentPayload]:
        import torch

        from difflet.models.hunyuan_video.application import HunyuanVideoDiTInputBundle

        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        request = invocation.request
        assert request.video is not None
        latent_frames = (request.video.num_frames - 1) // 4 + 1
        generator = torch.Generator().manual_seed(request.seed)
        latents = torch.randn(
            1,
            16,
            latent_frames,
            request.height // 8,
            request.width // 8,
            dtype=torch.bfloat16,
            generator=generator,
        )
        bundle = HunyuanVideoDiTInputBundle(
            hidden_states=latents,
            timestep=torch.zeros(1, dtype=torch.bfloat16),
            encoder_hidden_states=invocation.input.encoder_hidden_states,
            encoder_attention_mask=invocation.input.encoder_attention_mask,
            pooled_projections=invocation.input.pooled_projections,
            guidance=torch.full([1], request.guidance_scale * 1000.0, dtype=torch.bfloat16),
        )
        output = self.adapter._require_denoiser().pipeline(
            bundle=bundle,
            num_inference_steps=request.num_inference_steps,
            output_type="latent",
            return_trajectory=False,
        )
        invocation.context.cancellation.throw_if_cancelled()
        return stage_result(
            HunyuanVideoLatentPayload(output.latents),
            started_monotonic=started,
        )

    async def shutdown(self) -> None:
        return None


class HunyuanVideoHostDecoderStageRunner:
    def __init__(self, adapter: "HunyuanVideoServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[HunyuanVideoLatentPayload],
    ) -> StageExecutionResult[HunyuanVideoFinalPayload]:
        import torch

        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        vae = self.adapter._require_vae()
        scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
        latents = invocation.input.latents.detach().to(device="cpu", dtype=torch.float32)
        with torch.no_grad():
            frames = vae.decode(latents / scaling_factor, return_dict=False)[0]
        frames = frames.to(torch.float32).clamp(-1.0, 1.0)
        invocation.context.cancellation.throw_if_cancelled()
        output = encode_video_tensor(
            frames,
            invocation.request,
            layout="BCTHW",
            value_range="minus_one_to_one",
        )
        invocation.context.cancellation.throw_if_cancelled()
        return stage_result(HunyuanVideoFinalPayload(output), started_monotonic=started)

    async def shutdown(self) -> None:
        return None


class HunyuanVideoNeuronDecoderStageRunner:
    def __init__(self, adapter: "HunyuanVideoServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[HunyuanVideoLatentPayload],
    ) -> StageExecutionResult[HunyuanVideoFinalPayload]:
        import torch

        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        app = self.adapter._require_neuron_vae()
        latents = invocation.input.latents.detach().to(dtype=torch.bfloat16)
        # Reuse the lower-layer pipeline's scaling and segmented decode contract
        # instead of duplicating VAE normalization in the serving adapter.
        frames = app.pipeline._decode_latents(latents)
        frames = frames.to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)
        invocation.context.cancellation.throw_if_cancelled()
        output = encode_video_tensor(
            frames,
            invocation.request,
            layout="BCTHW",
            value_range="minus_one_to_one",
        )
        invocation.context.cancellation.throw_if_cancelled()
        return stage_result(HunyuanVideoFinalPayload(output), started_monotonic=started)

    async def shutdown(self) -> None:
        return None


class HunyuanVideoServingStageAdapter:
    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.runtime: ResolvedRuntimeBundle | None = None
        self.profile: ServingProfile | None = None
        self.clip_tokenizer = None
        self.clip_model = None
        self.clip_app = None
        self.llama_tokenizer = None
        self.llama_app = None
        self.denoiser = None
        self.vae = None
        self.vae_app = None
        self._smoke: StartupSmokeTarget | None = None

    async def create_loaded_runners(
        self,
        runtime: ResolvedRuntimeBundle,
    ) -> OrderedDict[str, ErasedStageRunner]:
        _validate_profile(runtime.profile)
        manager = ImmutableArtifactManager(
            runtime.profile.cache_dir or Path.home() / ".cache" / "difflet"
        )
        for binding in runtime.artifacts.bindings:
            spec = runtime.require_compile_spec(binding.artifact_id)
            manager.validate_binding(
                binding,
                validate_payload=lambda path, spec=spec: _validate_artifact(
                    runtime.source, runtime.profile, spec, path
                ),
            )
        clip_placement = _clip_placement(runtime.profile)
        if clip_placement == "host":
            self.clip_tokenizer, self.clip_model = _load_host_clip(runtime.source.pinned_model_path)
        else:
            self.clip_tokenizer = _load_clip_tokenizer(runtime.source.pinned_model_path)
        llama_binding = runtime.artifacts.require("llama")
        llama_app = _build_llama_app(
            runtime.source,
            runtime.profile,
            compiled_path=llama_binding.path,
            for_compile=False,
        )
        llama_app.load(str(llama_binding.path))
        self.llama_app = llama_app
        self.llama_tokenizer = _load_llama_tokenizer(runtime.source.pinned_model_path)
        denoiser_binding = runtime.artifacts.require("denoiser")
        denoiser = _build_denoiser(runtime.source, runtime.profile)
        denoiser.load(
            str(denoiser_binding.path),
            start_rank_id=0,
            local_ranks_size=runtime.profile.world_size,
            skip_warmup=True,
        )
        self.denoiser = denoiser
        if clip_placement == "neuron":
            # A full-world TP4 component must establish the process communicator
            # before replicated TP1/W4 components, matching the validated Flux
            # resident load order.
            clip_binding = runtime.artifacts.require("clip")
            clip_app = _build_clip_app(runtime.source, runtime.profile)
            clip_app.load(
                str(clip_binding.path),
                start_rank_id=0,
                local_ranks_size=runtime.profile.world_size,
                skip_warmup=True,
            )
            self.clip_app = clip_app
        if runtime.profile.host_vae:
            self.vae = _load_host_vae(runtime.source.pinned_model_path)
        else:
            decoder_binding = runtime.artifacts.require("decoder")
            vae_app = _build_vae_decoder(runtime.source, runtime.profile)
            vae_app.load(
                str(decoder_binding.path),
                start_rank_id=0,
                local_ranks_size=runtime.profile.world_size,
                skip_warmup=True,
            )
            self.vae_app = vae_app
        self.runtime = runtime
        self.profile = runtime.profile
        clip_runner = (
            HunyuanVideoHostClipStageRunner(self)
            if clip_placement == "host"
            else HunyuanVideoNeuronClipStageRunner(self)
        )
        decoder_runner = (
            HunyuanVideoHostDecoderStageRunner(self)
            if runtime.profile.host_vae
            else HunyuanVideoNeuronDecoderStageRunner(self)
        )
        return OrderedDict(
            (
                (
                    "clip",
                    ValidatedStageRunner(
                        clip_runner,
                        HunyuanVideoInitialPayload,
                        HunyuanVideoClipPayload,
                    ),
                ),
                (
                    "llama",
                    ValidatedStageRunner(
                        HunyuanVideoLlamaStageRunner(self),
                        HunyuanVideoClipPayload,
                        HunyuanVideoConditioningPayload,
                    ),
                ),
                (
                    "denoiser",
                    ValidatedStageRunner(
                        HunyuanVideoDenoiserStageRunner(self),
                        HunyuanVideoConditioningPayload,
                        HunyuanVideoLatentPayload,
                    ),
                ),
                (
                    "decoder",
                    ValidatedStageRunner(
                        decoder_runner,
                        HunyuanVideoLatentPayload,
                        HunyuanVideoFinalPayload,
                    ),
                ),
            )
        )

    def initial_payload(self, request: DiffletGenerateRequest) -> HunyuanVideoInitialPayload:
        require_video_target(request)
        return HunyuanVideoInitialPayload()

    def finalize(self, payload: StagePayload) -> FileBackedGenerateOutput:
        return require_exact_payload(
            payload,
            HunyuanVideoFinalPayload,
            boundary="HunyuanVideo final payload",
        ).output

    def smoke_request(self) -> DiffletGenerateRequest:
        if self.profile is None:
            raise RuntimeError("HunyuanVideo serving profile is not loaded")
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
            raise RuntimeError("HunyuanVideo serving profile is not loaded")
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
        self.vae_app = None
        self.denoiser = None
        self.llama_app = None
        self.llama_tokenizer = None
        self.clip_model = None
        self.clip_app = None
        self.clip_tokenizer = None
        self.profile = None
        self.runtime = None

    def _require_host_clip(self):
        if self.clip_tokenizer is None or self.clip_model is None:
            raise RuntimeError("HunyuanVideo host CLIP is not loaded")
        return self.clip_tokenizer, self.clip_model

    def _require_neuron_clip(self):
        if self.clip_tokenizer is None or self.clip_app is None:
            raise RuntimeError("HunyuanVideo Neuron CLIP is not loaded")
        return self.clip_tokenizer, self.clip_app

    def _require_llama(self):
        if self.llama_app is None or self.llama_tokenizer is None:
            raise RuntimeError("HunyuanVideo Llama is not loaded")
        return self.llama_app, self.llama_tokenizer

    def _require_denoiser(self):
        if self.denoiser is None:
            raise RuntimeError("HunyuanVideo denoiser is not loaded")
        return self.denoiser

    def _require_vae(self):
        if self.vae is None:
            raise RuntimeError("HunyuanVideo host VAE is not loaded")
        return self.vae

    def _require_neuron_vae(self):
        if self.vae_app is None:
            raise RuntimeError("HunyuanVideo Neuron VAE is not loaded")
        return self.vae_app


def _validate_profile(profile: ServingProfile) -> None:
    if profile.model_type != _MODEL_TYPE or profile.model_id != _HF_MODEL_ID:
        raise ValueError("HunyuanVideo serving adapter only supports the community 1.0 checkpoint")
    if profile.output_modality != "video" or profile.output_mime_type != "video/mp4":
        raise ValueError("HunyuanVideo serving profile must produce video/mp4")
    if profile.dtype.lower().removeprefix("torch.") not in {"bf16", "bfloat16"}:
        raise ValueError("HunyuanVideo resident serving requires bfloat16")
    spatial_multiple = _VAE_SPATIAL_SCALE * _PATCH_SIZE
    for value, name in ((profile.height, "height"), (profile.width, "width")):
        if type(value) is not int or value <= 0 or value % spatial_multiple:
            raise ValueError(
                f"HunyuanVideo serving {name} must be a positive integer divisible by "
                f"{spatial_multiple}"
            )
    if type(profile.num_frames) is not int or profile.num_frames <= 0:
        raise ValueError("HunyuanVideo serving num_frames must be a positive integer")
    if (profile.num_frames - 1) % _VAE_TEMPORAL_SCALE:
        raise ValueError(
            "HunyuanVideo serving num_frames must equal 4n+1 for exact causal VAE reconstruction"
        )
    if profile.clip_placement not in {None, "host", "neuron"}:
        raise ValueError("HunyuanVideo CLIP placement must be host or neuron")
    if profile.output_fps != _FPS:
        raise ValueError(f"HunyuanVideo video profile requires {_FPS} FPS")
    parallel = profile.parallel
    if parallel.cp_degree != 1:
        raise ValueError("HunyuanVideo resident serving requires cp_degree=1")
    if parallel.cfg_parallel_enabled:
        raise ValueError("HunyuanVideo resident serving requires CFG parallel off")
    if parallel.tp_degree != _TP_DEGREE:
        raise ValueError("HunyuanVideo resident serving requires tp_degree=4")
    if parallel.dp_degree != 1:
        raise ValueError("HunyuanVideo resident serving requires dp_degree=1")
    if profile.world_size != _WORLD_SIZE:
        raise ValueError("HunyuanVideo resident serving requires world_size=4")
    if profile.teacache_speedup is not None or profile.teacache_calibration_data is not None:
        raise ValueError("HunyuanVideo resident video serving does not support TeaCache")


def _compile_specs(
    source: ResolvedModelSource,
    profile: ServingProfile,
) -> tuple[DiffletCompileSpec, ...]:
    # v2: denoiser/decoder identities carry the canonical shape SET (K=1 uses
    # the same list form) instead of a single height/width/num_frames.
    common = {
        "compile_contract_version": 2,
        "model_type": _MODEL_TYPE,
        "model_id": source.model_id,
        "resolved_source_id": source.resolved_source_id,
        "tp_degree": profile.parallel.tp_degree,
        "cp_degree": profile.parallel.cp_degree,
        "sp_enabled": profile.parallel.sp_enabled,
        "world_size": profile.world_size,
        "dtype": profile.dtype,
        "virtual_core_size": _VIRTUAL_CORE_SIZE,
        "toolchain": toolchain_versions(),
    }
    from difflet.backends.trainium.core.bucketing import canonicalize_shapes

    profile_shapes = getattr(profile, "shapes", None) or (
        (profile.height, profile.width, profile.num_frames),
    )
    shapes_list = [list(shape) for shape in canonicalize_shapes(profile_shapes)]
    llama_identity = CompileArtifactIdentity.from_cache_inputs(
        {
            **common,
            "component_id": "llama",
            "sequence_length": _LLAMA_SEQ_LEN,
            "tensor_capture": _LLAMA_CAPTURE,
        }
    )
    denoiser_identity = CompileArtifactIdentity.from_cache_inputs(
        {
            **common,
            "component_id": "denoiser",
            "shapes": shapes_list,
            "text_seq_len": _TEXT_SEQ_LEN,
        }
    )
    specs: list[DiffletCompileSpec] = []
    if _clip_placement(profile) == "neuron":
        clip_identity = CompileArtifactIdentity.from_cache_inputs(
            {
                **common,
                "component_id": "clip",
                "sequence_length": 77,
                "component_tp_degree": 1,
                "component_world_size": profile.world_size,
            }
        )
        specs.append(DiffletCompileSpec("clip", "clip", clip_identity))
    specs.extend(
        (
            DiffletCompileSpec("llama", "llama", llama_identity),
            DiffletCompileSpec("denoiser", "denoiser", denoiser_identity),
        )
    )
    if not profile.host_vae:
        decoder_identity = CompileArtifactIdentity.from_cache_inputs(
            {
                **common,
                "component_id": "decoder",
                "shapes": shapes_list,
                "component_tp_degree": 1,
                "component_world_size": profile.world_size,
                "segmented_causal_norm_conv": True,
            }
        )
        specs.append(DiffletCompileSpec("decoder", "decoder", decoder_identity))
    return tuple(specs)


def _compile_artifact(
    source: ResolvedModelSource,
    profile: ServingProfile,
    spec: DiffletCompileSpec,
    target: ArtifactPublishTarget,
) -> None:
    if target.artifact_id != spec.artifact_id or target.identity != spec.identity:
        raise ValueError("HunyuanVideo compile target does not match compile spec")
    with serving_compile_environment(profile.world_size, virtual_core_size=_VIRTUAL_CORE_SIZE):
        if spec.component_id == "clip":
            app = _build_clip_app(source, profile)
            app.compile(str(target.staging_path))
        elif spec.component_id == "llama":
            app = _build_llama_app(
                source,
                profile,
                compiled_path=target.staging_path,
                for_compile=True,
            )
            app.compile(str(target.staging_path))
        elif spec.component_id == "denoiser":
            app = _build_denoiser(source, profile)
            app.compile(str(target.staging_path))
        elif spec.component_id == "decoder":
            app = _build_vae_decoder(source, profile)
            app.compile(str(target.staging_path))
        else:
            raise ValueError(f"unknown HunyuanVideo component {spec.component_id!r}")
    _validate_artifact(source, profile, spec, target.staging_path)


def _validate_artifact(
    source: ResolvedModelSource,
    profile: ServingProfile,
    spec: DiffletCompileSpec,
    path: Path,
) -> None:
    if spec.component_id == "clip":
        if not _has_nxd_component(path):
            raise ValueError(f"HunyuanVideo CLIP artifact is incomplete at {path}")
        return
    if spec.component_id == "llama":
        if not _has_nxd_component(path):
            raise ValueError(f"HunyuanVideo Llama artifact is incomplete at {path}")
        return
    if spec.component_id == "denoiser":
        app = _build_denoiser(source, profile)
        if not app.has_compiled_artifacts(str(path)) or not compiled_model_payloads_ready(
            app, path
        ):
            raise ValueError(f"HunyuanVideo denoiser artifact is incomplete at {path}")
        return
    if spec.component_id == "decoder":
        app = _build_vae_decoder(source, profile)
        if not app.has_compiled_artifacts(str(path)) or not compiled_model_payloads_ready(
            app, path
        ):
            raise ValueError(f"HunyuanVideo decoder artifact is incomplete at {path}")
        return
    raise ValueError(f"unknown HunyuanVideo component {spec.component_id!r}")


def _build_llama_app(
    source: ResolvedModelSource,
    profile: ServingProfile,
    *,
    compiled_path: Path,
    for_compile: bool,
):
    import torch
    from neuronx_distributed_inference.models.config import NeuronConfig, TensorCaptureConfig
    from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from transformers import AutoConfig

    encoder_path = str(Path(source.pinned_model_path) / "text_encoder")
    hf_config = AutoConfig.from_pretrained(encoder_path)
    if hf_config.pad_token_id is None:
        hf_config.pad_token_id = 0
    hf_config.tie_word_embeddings = True
    weights_dir = Path(compiled_path) / "weights"
    shard_paths = [
        weights_dir / f"tp{rank}_sharded_checkpoint.safetensors"
        for rank in range(profile.parallel.tp_degree)
    ]
    neuron_config = NeuronConfig(
        tp_degree=profile.parallel.tp_degree,
        batch_size=1,
        seq_len=_LLAMA_SEQ_LEN,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config={},
        tensor_capture_config=TensorCaptureConfig(modules_to_capture=[_LLAMA_CAPTURE]),
        save_sharded_checkpoint=for_compile or all(path.exists() for path in shard_paths),
    )
    config = NeuronLlamaForCausalLM.get_config_cls()(
        neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )
    return NeuronLlamaForCausalLM(encoder_path, config)


def _build_denoiser(source: ResolvedModelSource, profile: ServingProfile):
    from difflet.models.hunyuan_video.application import NeuronHunyuanVideoApplication

    app = NeuronHunyuanVideoApplication(
        model_path=source.pinned_model_path,
        parallel=profile.parallel,
        dtype=_torch_bfloat16(),
        shape=profile.shape_dict(),
        text_seq_len=_TEXT_SEQ_LEN,
        enable_transformer=True,
        enable_vae_decoder=False,
    )
    # The current CLI does not compile/load the optional TeaCache probe for its
    # baseline generation artifact; preserve that exact lower-layer contract.
    app.teacache_probe = None
    return app


def _build_clip_app(source: ResolvedModelSource, profile: ServingProfile):
    import torch

    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.models.flux.clip.modeling_clip import (
        CLIPInferenceConfig,
        NeuronClipApplication,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    clip_path = str(Path(source.pinned_model_path) / "text_encoder_2")
    config = CLIPInferenceConfig(
        neuron_config=NeuronConfig(
            tp_degree=1,
            world_size=profile.world_size,
            torch_dtype=torch.bfloat16,
        ),
        load_config=load_diffusers_config(clip_path),
    )
    for key, value in {
        "output_attentions": False,
        "output_hidden_states": False,
        "use_return_dict": True,
    }.items():
        setattr(config, key, value)
    return NeuronClipApplication(model_path=clip_path, config=config)


def _build_vae_decoder(source: ResolvedModelSource, profile: ServingProfile):
    from difflet.models.hunyuan_video.application import NeuronHunyuanVideoApplication

    return NeuronHunyuanVideoApplication(
        model_path=source.pinned_model_path,
        parallel=profile.parallel,
        dtype=_torch_bfloat16(),
        shape=profile.shape_dict(),
        text_seq_len=_TEXT_SEQ_LEN,
        enable_transformer=False,
        enable_vae_decoder=True,
    )


def _runtime_plan(
    profile: ServingProfile,
    specs: tuple[DiffletCompileSpec, ...],
) -> RuntimePlan:
    environment, allocation = resident_environment(
        profile,
        allocation_id="hunyuan-video-resident",
        virtual_core_size=_VIRTUAL_CORE_SIZE,
    )
    topology = ParallelTopology(
        tp_degree=profile.parallel.tp_degree,
        cp_degree=profile.parallel.cp_degree,
        world_size=profile.world_size,
    )
    replicated_topology = ParallelTopology(
        tp_degree=1,
        cp_degree=1,
        world_size=profile.world_size,
    )
    clip_placement = _clip_placement(profile)
    return RuntimePlan(
        mode="resident",
        profile_identity=combined_profile_identity(
            f"clip:{clip_placement}",
            *(spec.identity.digest for spec in specs),
            f"vae:{profile.vae_placement}",
        ),
        environment=environment,
        allocations=(allocation,),
        stages=(
            (
                StageRuntimeSpec("clip", None, None, None, placement="host")
                if clip_placement == "host"
                else StageRuntimeSpec("clip", allocation.allocation_id, replicated_topology, "clip")
            ),
            StageRuntimeSpec("llama", allocation.allocation_id, topology, "llama"),
            StageRuntimeSpec("denoiser", allocation.allocation_id, topology, "denoiser"),
            (
                StageRuntimeSpec("decoder", None, None, None, placement="host")
                if profile.host_vae
                else StageRuntimeSpec(
                    "decoder", allocation.allocation_id, replicated_topology, "decoder"
                )
            ),
        ),
    )


def _load_host_clip(model_path: str):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    path = str(Path(model_path) / "text_encoder_2")
    tokenizer = CLIPTokenizer.from_pretrained(str(Path(model_path) / "tokenizer_2"))
    model = CLIPTextModel.from_pretrained(path, torch_dtype=torch.float32).eval()
    return tokenizer, model


def _load_clip_tokenizer(model_path: str):
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained(str(Path(model_path) / "tokenizer_2"))


def _load_llama_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(Path(model_path) / "tokenizer"))


def _load_host_vae(model_path: str):
    import torch
    from diffusers import AutoencoderKLHunyuanVideo

    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        str(Path(model_path) / "vae"), torch_dtype=torch.float32
    ).eval()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    return vae


def _has_nxd_component(path: Path) -> bool:
    model = path / "model.pt"
    config = path / "neuron_config.json"
    return (
        model.is_file()
        and model.stat().st_size > 0
        and config.is_file()
        and config.stat().st_size > 0
    )


def _pipeline_definition(profile: ServingProfile | None = None) -> PipelineDefinition:
    from difflet.common.registry.hunyuan_video import serving_metadata

    pipeline = serving_metadata().pipeline_definition
    if profile is None:
        return pipeline
    clip_runner = (
        "HunyuanVideoHostClipStageRunner"
        if _clip_placement(profile) == "host"
        else "HunyuanVideoNeuronClipStageRunner"
    )
    decoder_runner = (
        "HunyuanVideoHostDecoderStageRunner"
        if profile.host_vae
        else "HunyuanVideoNeuronDecoderStageRunner"
    )
    module = __name__
    stages = list(pipeline.stages)
    stages[0] = replace(stages[0], runner_factory=f"{module}:{clip_runner}")
    stages[-1] = replace(stages[-1], runner_factory=f"{module}:{decoder_runner}")
    return PipelineDefinition(model_type=pipeline.model_type, stages=tuple(stages))


def _clip_placement(profile: ServingProfile) -> str:
    return profile.clip_placement or "host"


def _torch_bfloat16():
    import torch

    return torch.bfloat16


__all__ = [
    "HunyuanVideoDenoiserStageRunner",
    "HunyuanVideoHostClipStageRunner",
    "HunyuanVideoHostDecoderStageRunner",
    "HunyuanVideoNeuronClipStageRunner",
    "HunyuanVideoNeuronDecoderStageRunner",
    "HunyuanVideoLlamaStageRunner",
    "HunyuanVideoServingArtifactPreparer",
    "HunyuanVideoServingRequestValidator",
    "HunyuanVideoServingStageAdapter",
]
