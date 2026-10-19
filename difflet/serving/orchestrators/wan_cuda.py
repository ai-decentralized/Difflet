"""Wan 2.2 T2V-A14B served on the CUDA backend — one resident process, one stage.

``qwen_image_cuda``'s shape for video: at ``tp=1`` the whole model — UMT5, both experts,
the VAE — is one ``CudaWanApplication`` behind ``DiffletPipeline``, so the resident worker gets
a single opaque ``pipeline`` stage that holds ``DiffletPipeline.from_pretrained(backend="cuda")``
and calls ``app.generate()`` per request with the CUDA CLI's kwargs
(``difflet/cli/orchestrators/wan_cuda.py``). The ``(B, 3, F, H, W)`` frames in [-1, 1] go to the
request's MP4 target through the serving layer's ``encode_video_tensor``; ``/v1/videos`` and
``/v1/videos/sync`` deliver it, as on Trainium. The engine, the HTTP layer and the Trainium Wan
adapter (``difflet/serving/models/wan.py``) are untouched. ``--torch-compile`` reaches the
application as it does on the CLI.

The request contract: MP4 with the resident profile's (height, width, num_frames) and fps;
``guidance_scale`` (the high-noise expert) and ``guidance_scale_2`` (the low-noise expert) in
[0, 20]; an optional ``negative_prompt``; both prompts within UMT5's 512 tokens (diffusers would
truncate silently). ``boundary_ratio``, ``flow_shift`` and ``true_cfg_scale`` are refused: the
CUDA application takes the checkpoint's expert boundary and scheduler shift and runs true CFG
already. Release 1 scope: tp=1, cp=1, one shape, no TeaCache, the VAE on the GPU; anything
wider is refused in ``prepare_runtime``, before the 126 GB snapshot is resolved.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from difflet.serving.models._common import (
    StartupSmokeTarget,
    encode_video_tensor,
    validate_smoke_output,
)
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators import cuda_resident
from difflet.serving.orchestrators.base import resolve_hf_model_source, validate_guidance_scale
from difflet.serving.types import (
    ArtifactSet,
    DiffletGenerateRequest,
    FileBackedGenerateOutput,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    RuntimePlan,
    ServingProfile,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
    VideoGenerateOptions,
)

_HF_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
_MODEL_TYPE = "wan"
_ALLOCATION_ID = "wan-cuda"
_STAGE_ID = cuda_resident.STAGE_ID
_TAG = "[wan/cuda serve]"
#: Wan 2.2's high-noise guidance default (the CUDA CLI's), the startup smoke's value.
DEFAULT_GUIDANCE_SCALE = 4.0
MAX_GUIDANCE_SCALE = 20.0  # the Trainium Wan validator's bound, for both experts
TEXT_SEQ_LEN = 512  # UMT5 tokens (WanPipeline's max_sequence_length)
#: Startup smoke: four steps run both experts (with the checkpoint's flow shift they are about
#: 999 / 900 / 750 / 500, two on each side of Wan 2.2's 0.875 boundary), so a broken second
#: expert fails startup, not a request.
SMOKE_STEPS = 4
#: Video fields the CUDA application does not take, with why.
_REFUSED_VIDEO_FIELDS = {
    "boundary_ratio": "the expert boundary is the checkpoint's (model_index.json)",
    "flow_shift": "the scheduler shift is the checkpoint's",
    "true_cfg_scale": "Wan already runs true CFG; use guidance_scale and guidance_scale_2",
}


@dataclass(frozen=True, slots=True)
class CudaWanInitialPayload(StagePayload):
    pass


@dataclass(frozen=True, slots=True)
class CudaWanFinalPayload(StagePayload):
    output: FileBackedGenerateOutput


def refuse_unsupported_profile(profile: ServingProfile) -> None:
    """Raise ``feature_not_supported`` for every profile the CUDA backend does not serve.

    ``build_serving_profile`` already refuses ``--cfg-parallel`` and the TeaCache cadence /
    online-delta modes for every serving model; the checks here are the worker's backstop for a
    bundle that did not come through it."""
    cuda_resident.refuse_shared_profile(profile, tag=_TAG)
    if profile.parallel.cfg_parallel_enabled:
        cuda_resident.refuse(
            _TAG, "--cfg-parallel (one CFG branch per GPU, a `difflet run` feature)"
        )
    if profile.teacache_speedup is not None:
        cuda_resident.refuse(_TAG, "--teacache-speedup (TeaCache for Wan)")
    if profile.host_vae:
        cuda_resident.refuse(_TAG, "--host-vae (the VAE already runs on the GPU)")


def cuda_runtime_plan(profile: ServingProfile, source: ResolvedModelSource) -> RuntimePlan:
    """The resident plan for one GPU: one allocation (slot 0), one eager ``cuda`` stage."""
    return cuda_resident.resident_runtime_plan(
        profile, source, allocation_id=_ALLOCATION_ID, stage_id=_STAGE_ID
    )


class CudaWanServingArtifactPreparer:
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
        refuse_unsupported_profile(profile)
        print(f"[difflet serve] resolving Wan 2.2 weights for {self.model_id}")
        source = resolve_hf_model_source(
            self.model_id, revision=self.revision, download_policy=download_policy
        )
        print("[difflet serve] cuda backend: eager resident pipeline, no compile artifacts")
        from difflet.common.registry.wan import cuda_serving_metadata

        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=cuda_serving_metadata().pipeline_definition,
            runtime_plan=cuda_runtime_plan(profile, source),
            compile_specs=(),
            artifacts=ArtifactSet(()),
            adapter_config={"compile_policy": compile_policy},
        )


def _video(request: DiffletGenerateRequest) -> VideoGenerateOptions:
    if request.output_format != "mp4":
        raise invalid_extra_body("Wan serving on CUDA writes MP4 only")
    if request.video is None:
        raise invalid_extra_body("a Wan request needs its video options (num_frames, fps)")
    return request.video


class CudaWanServingRequestValidator:
    """The request contract, enforced in the server process before the worker."""

    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, runtime: ResolvedRuntimeBundle) -> None:
        self.model_id = runtime.profile.model_id
        self.runtime = runtime
        self._tokenizer: Any = None

    def preload(self) -> None:
        self._tokenizer_for_runtime()

    def validate(self, request: DiffletGenerateRequest) -> None:
        video = _video(request)
        profile = self.runtime.profile
        requested = (request.height, request.width, video.num_frames)
        resident = (profile.height, profile.width, profile.num_frames)
        if requested != resident:
            raise profile_mismatch(
                "request video shape {}x{}x{} is not the resident CUDA profile shape "
                "{}x{}x{}".format(*requested, *resident)
            )
        if video.fps != profile.output_fps:
            raise profile_mismatch(
                f"request fps {video.fps} is not the resident CUDA profile's {profile.output_fps}"
            )
        validate_guidance_scale(request, maximum=MAX_GUIDANCE_SCALE)
        g2 = video.guidance_scale_2
        if g2 is not None and not 0 <= float(g2) <= MAX_GUIDANCE_SCALE:
            raise invalid_extra_body(
                f"guidance_scale_2 must satisfy 0 <= value <= {MAX_GUIDANCE_SCALE:g}"
            )
        for field, why in _REFUSED_VIDEO_FIELDS.items():
            if getattr(video, field) is not None:
                raise feature_not_supported(
                    f"{_TAG} {field} is not taken by Wan 2.2 on the CUDA backend: {why}"
                )
        for name, text in (("prompt", request.prompt), ("negative_prompt", video.negative_prompt)):
            if text is None:
                continue
            encoded = self._tokenizer_for_runtime()(
                text, padding=False, truncation=False, return_tensors="pt"
            )
            if len(encoded.input_ids[0]) > TEXT_SEQ_LEN:
                raise prompt_too_long(f"Wan {name} exceeds UMT5's {TEXT_SEQ_LEN} tokens")

    def _tokenizer_for_runtime(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_dir = self.runtime.source.pinned_model_path
            self._tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
        return self._tokenizer


class CudaWanPipelineRunner:
    """The one stage: prompt in, MP4 out, through the resident ``DiffletPipeline``."""

    def __init__(self, adapter: "CudaWanServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[CudaWanInitialPayload],
    ) -> StageExecutionResult[CudaWanFinalPayload]:
        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        frames = self.adapter.generate_frames(invocation.request)
        invocation.context.cancellation.throw_if_cancelled()
        output = encode_video_tensor(
            frames, invocation.request, layout="BCTHW", value_range="minus_one_to_one"
        )
        return stage_result(CudaWanFinalPayload(output=output), started_monotonic=started)

    async def shutdown(self) -> None:
        self.adapter.pipe = None


class CudaWanServingStageAdapter:
    """``ServingStageAdapter`` for Wan 2.2 on CUDA: the model stays resident in the worker."""

    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.active_runtime: ResolvedRuntimeBundle | None = None
        self.active_profile: ServingProfile | None = None
        self.pipe: Any = None
        self._smoke: StartupSmokeTarget | None = None

    async def create_loaded_runners(
        self,
        runtime: ResolvedRuntimeBundle,
    ) -> OrderedDict[str, ErasedStageRunner]:
        profile = runtime.profile
        refuse_unsupported_profile(profile)
        # Worker-side imports: the parent process never loads torch for this adapter.
        from difflet.cli.orchestrators.qwen_image_cuda import cuda_pipeline_kwargs
        from difflet.pipeline.difflet_pipeline import DiffletPipeline

        config = runtime.adapter_config or {}
        print("[difflet serve] loading Wan 2.2 on CUDA (UMT5, both experts, VAE in-process)")
        self.active_runtime = runtime
        self.active_profile = profile
        # The CLI's kwargs (WanCudaOrchestrator.pipeline_kwargs: the shared ones with a video's
        # shape); the revision is the commit the parent pinned.
        kwargs = cuda_pipeline_kwargs(
            tp_degree=profile.parallel.tp_degree,
            height=profile.height,
            width=profile.width,
            cache_dir=profile.cache_dir,
            revision=runtime.source.resolved_source_id,
            force_compile=config.get("compile_policy") is CompilePolicy.FORCE,
            load=True,
            torch_compile=profile.torch_compile,
        )
        kwargs["num_frames"] = int(profile.num_frames)
        self.pipe = DiffletPipeline.from_pretrained(self.model_id, **kwargs)
        print("[difflet serve] Wan 2.2 CUDA pipeline resident")
        return OrderedDict(
            (
                (
                    _STAGE_ID,
                    ValidatedStageRunner(
                        CudaWanPipelineRunner(self), CudaWanInitialPayload, CudaWanFinalPayload
                    ),
                ),
            )
        )

    def generate_frames(self, request: DiffletGenerateRequest):
        """``app.generate`` with the CLI's arguments (``WanCudaOrchestrator.generate_kwargs``),
        plus ``guidance_scale_2`` and ``negative_prompt`` when the request carries them."""
        if self.pipe is None or self.active_profile is None:
            raise RuntimeError("Wan 2.2 CUDA pipeline is not loaded")
        import torch

        video = _video(request)
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

    def initial_payload(self, request: DiffletGenerateRequest) -> CudaWanInitialPayload:
        return CudaWanInitialPayload()

    def finalize(self, payload: StagePayload) -> FileBackedGenerateOutput:
        return require_exact_payload(
            payload, CudaWanFinalPayload, boundary="Wan CUDA final payload"
        ).output

    def smoke_request(self) -> DiffletGenerateRequest:
        if self.pipe is None or self.active_profile is None:
            raise RuntimeError("Wan 2.2 CUDA load did not leave a resident pipeline")
        if self._smoke is not None:
            self._smoke.cleanup()
        self._smoke = StartupSmokeTarget()
        profile = self.active_profile
        return DiffletGenerateRequest(
            request_id="startup-smoke",
            model=self.model_id,
            prompt="a small red square",
            height=profile.height,
            width=profile.width,
            num_inference_steps=SMOKE_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            seed=0,
            output_format="mp4",
            video=VideoGenerateOptions(
                num_frames=int(profile.num_frames),
                fps=int(profile.output_fps),
                output_target=self._smoke.file_target(),
            ),
        )

    def validate_smoke_output(self, output: FileBackedGenerateOutput) -> None:
        if self.active_profile is None or self._smoke is None:
            raise RuntimeError("Wan 2.2 CUDA smoke ran without a profile or a target")
        try:
            validate_smoke_output(
                output, profile=self.active_profile, expected_path=self._smoke.path
            )
        finally:
            self._smoke.cleanup()
            self._smoke = None
        print("[difflet serve] Wan 2.2 CUDA generation smoke passed")

    def reset_request_state(self, outcome: str) -> None:
        return None

    async def shutdown(self) -> None:
        if self._smoke is not None:
            self._smoke.cleanup()
            self._smoke = None
        self.pipe = None
        self.active_profile = None
        self.active_runtime = None


__all__: list[str] = [
    "DEFAULT_GUIDANCE_SCALE",
    "MAX_GUIDANCE_SCALE",
    "SMOKE_STEPS",
    "TEXT_SEQ_LEN",
    "CudaWanFinalPayload",
    "CudaWanInitialPayload",
    "CudaWanPipelineRunner",
    "CudaWanServingArtifactPreparer",
    "CudaWanServingRequestValidator",
    "CudaWanServingStageAdapter",
    "cuda_runtime_plan",
    "refuse_unsupported_profile",
]
