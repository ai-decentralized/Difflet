"""Qwen-Image served on the CUDA backend — one resident process, one stage.

The Trainium adapter (``qwen_image.py``) fills the serving engine's resident worker with three
Neuron stage applications and a NeuronCore allocation. On CUDA at ``tp=1`` the whole model is one
``CudaQwenImageApplication`` behind ``DiffletPipeline``, so the worker gets a single opaque
``pipeline`` stage that holds ``DiffletPipeline.from_pretrained(backend="cuda")`` resident and
calls ``app.generate()`` per request — the same two calls ``difflet run`` makes on CUDA
(``difflet/cli/orchestrators/qwen_image_cuda.py``). Selected by backend name in
``difflet.serving.model_registry`` (``cuda_serving_metadata``); the engine, the HTTP layer and
the Trainium adapter are untouched.

Release 1 scope: tp=1, cp=1, dp=1, no CFG-parallel, no TeaCache, one request shape. Anything
wider is refused in ``prepare_runtime`` — before the 54 GB snapshot is resolved.

The resident plan (slot ``0``, one ``cuda`` stage) and the refusals every CUDA profile shares
live in ``cuda_resident``, shared with the Flux and Wan adapters.
"""

from __future__ import annotations

import io
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from difflet.common.orchestrators.qwen_image import TEXT_SEQ_LEN
from difflet.serving.engines.stage_pipeline import (
    ErasedStageRunner,
    ValidatedStageRunner,
    require_exact_payload,
    stage_result,
)
from difflet.serving.errors import feature_not_supported, profile_mismatch, prompt_too_long
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators import cuda_resident
from difflet.serving.orchestrators.base import resolve_hf_model_source
from difflet.serving.types import (
    ArtifactSet,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    QwenFinalPayload,
    QwenInitialPayload,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    RuntimePlan,
    ServingProfile,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
)

_HF_MODEL_ID = "Qwen/Qwen-Image"
_MODEL_TYPE = "qwen_image"
_ALLOCATION_ID = "qwen-cuda"
_STAGE_ID = "pipeline"
_TAG = "[qwen_image/cuda serve]"
#: The value the CUDA application fills into Qwen-Image's ignored guidance tensor
#: (``guidance_embeds=False``); the serving default, and the only value a request may carry.
NOOP_GUIDANCE_SCALE = 4.0
# diffusers' QwenImagePipeline prompt template and the prefix length it drops; the encoder
# truncates at tokenizer_max_length (TEXT_SEQ_LEN) + the prefix, silently.
_QWEN_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_QWEN_DROP_IDX = 34
#: Startup smoke: cheap, like the Trainium adapter's 4-step smoke; proves load + one generate.
SMOKE_STEPS = 4


def _refuse(what: str) -> None:
    cuda_resident.refuse(_TAG, what)


def refuse_unsupported_profile(profile: ServingProfile) -> None:
    """Raise ``feature_not_supported`` for every profile the CUDA backend does not serve."""
    cuda_resident.refuse_shared_profile(profile, tag=_TAG)
    if profile.parallel.cfg_parallel_enabled:
        # There is no second branch to place on another GPU.
        raise feature_not_supported(
            f"{_TAG} --cfg-parallel does not apply to Qwen-Image on the CUDA backend: "
            "it is a guidance-distilled/single-pass model here, no CFG "
            "(guidance_embeds=False, one transformer pass per step)"
        )
    if profile.teacache_speedup is not None:
        _refuse("--teacache-speedup (TeaCache port)")


def cuda_runtime_plan(profile: ServingProfile, source: ResolvedModelSource) -> RuntimePlan:
    """The resident plan for one GPU: one allocation (slot 0), one eager ``cuda`` stage."""
    return cuda_resident.resident_runtime_plan(
        profile, source, allocation_id=_ALLOCATION_ID, stage_id=_STAGE_ID
    )


class CudaQwenImageServingArtifactPreparer:
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
        print(f"[difflet serve] resolving Qwen-Image weights for {self.model_id}")
        source = resolve_hf_model_source(
            self.model_id, revision=self.revision, download_policy=download_policy
        )
        # No compile phase: the CUDA backend is eager, and the manifest-only
        # artifact is written by DiffletPipeline inside the worker at load. An empty
        # spec/binding pair keeps ResolvedRuntimeBundle's invariant true, as on TPU.
        print("[difflet serve] cuda backend: eager resident pipeline, no compile artifacts")
        from difflet.common.registry.qwen_image import cuda_serving_metadata

        pipeline = cuda_serving_metadata().pipeline_definition
        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=pipeline,
            runtime_plan=cuda_runtime_plan(profile, source),
            compile_specs=(),
            artifacts=ArtifactSet(()),
            adapter_config={"compile_policy": compile_policy},
        )


class CudaQwenImageServingRequestValidator:
    """The CUDA CLI's request contract, enforced in the server process before the worker."""

    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, runtime: ResolvedRuntimeBundle) -> None:
        self.model_id = runtime.profile.model_id
        self.runtime = runtime
        self._tokenizer: Any = None

    def preload(self) -> None:
        self._tokenizer_for_runtime()

    def validate(self, request: DiffletGenerateRequest) -> None:
        if float(request.guidance_scale) != NOOP_GUIDANCE_SCALE:
            raise feature_not_supported(
                "guidance_scale has no effect on Qwen-Image (guidance_embeds=False); Release 1 "
                "generation on CUDA is single-pass without classifier-free guidance — omit it"
            )
        profile = self.runtime.profile
        if (request.height, request.width) != (profile.height, profile.width):
            raise profile_mismatch(
                f"request shape {request.height}x{request.width} is not the resident CUDA "
                f"profile shape {profile.height}x{profile.width}"
            )
        encoded = self._tokenizer_for_runtime()(
            _QWEN_TEMPLATE.format(request.prompt),
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        if len(encoded.input_ids[0]) - _QWEN_DROP_IDX > TEXT_SEQ_LEN:
            raise prompt_too_long(f"Qwen prompt exceeds the DiT text length {TEXT_SEQ_LEN}")

    def _tokenizer_for_runtime(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_dir = self.runtime.source.pinned_model_path
            self._tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
        return self._tokenizer


class CudaQwenImagePipelineRunner:
    """The one stage: prompt in, PNG out, through the resident ``DiffletPipeline``."""

    def __init__(self, adapter: "CudaQwenImageServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[QwenInitialPayload],
    ) -> StageExecutionResult[QwenFinalPayload]:
        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        output = self.adapter.generate_png(invocation.request)
        return stage_result(QwenFinalPayload(output=output), started_monotonic=started)

    async def shutdown(self) -> None:
        self.adapter.pipe = None


class CudaQwenImageServingStageAdapter:
    """``ServingStageAdapter`` for Qwen-Image on CUDA: the model stays resident in the worker."""

    model_id = _HF_MODEL_ID
    model_type = _MODEL_TYPE

    def __init__(self, model_id: str = _HF_MODEL_ID) -> None:
        self.model_id = model_id
        self.active_runtime: ResolvedRuntimeBundle | None = None
        self.active_profile: ServingProfile | None = None
        self.pipe: Any = None

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
        print("[difflet serve] loading Qwen-Image on CUDA (text encoder, DiT, VAE in-process)")
        self.active_runtime = runtime
        self.active_profile = profile
        self.pipe = DiffletPipeline.from_pretrained(
            self.model_id,
            **cuda_pipeline_kwargs(
                tp_degree=profile.parallel.tp_degree,
                height=profile.height,
                width=profile.width,
                cache_dir=profile.cache_dir,
                # The commit the parent pinned (resolve_hf_model_source), not the user's ref:
                # a worker restart must load the snapshot the bundle and the validator's
                # tokenizer describe, even if refs/main moved meanwhile.
                revision=runtime.source.resolved_source_id,
                force_compile=config.get("compile_policy") is CompilePolicy.FORCE,
                load=True,
                torch_compile=profile.torch_compile,
            ),
        )
        print("[difflet serve] Qwen-Image CUDA pipeline resident")
        return OrderedDict(
            (
                (
                    _STAGE_ID,
                    ValidatedStageRunner(
                        CudaQwenImagePipelineRunner(self), QwenInitialPayload, QwenFinalPayload
                    ),
                ),
            )
        )

    def generate_png(self, request: DiffletGenerateRequest) -> DiffletGenerateOutput:
        """``app.generate`` with the CLI's arguments; no ``guidance_scale`` (the app refuses it)."""
        if self.pipe is None:
            raise RuntimeError("Qwen-Image CUDA pipeline is not loaded")
        image = self.pipe.app.generate(
            request.prompt,
            num_inference_steps=int(request.num_inference_steps),
            seed=int(request.seed),
            output_type="pil",
        )[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return DiffletGenerateOutput(
            data=buf.getvalue(), mime_type="image/png", output_format="png"
        )

    def initial_payload(self, request: DiffletGenerateRequest) -> QwenInitialPayload:
        return QwenInitialPayload()

    def finalize(self, payload: StagePayload) -> DiffletGenerateOutput:
        return require_exact_payload(
            payload, QwenFinalPayload, boundary="Qwen CUDA final payload"
        ).output

    def smoke_request(self) -> DiffletGenerateRequest:
        if self.pipe is None or self.active_profile is None:
            raise RuntimeError("Qwen-Image CUDA load did not leave a resident pipeline")
        profile = self.active_profile
        return DiffletGenerateRequest(
            request_id="startup-smoke",
            model=self.model_id,
            prompt="a small red square",
            height=profile.height,
            width=profile.width,
            num_inference_steps=SMOKE_STEPS,
            guidance_scale=NOOP_GUIDANCE_SCALE,
            seed=0,
        )

    def validate_smoke_output(self, output: DiffletGenerateOutput) -> None:
        if not output.data:
            raise RuntimeError("Qwen-Image CUDA smoke produced empty output")
        from PIL import Image

        profile = self.active_profile
        size = Image.open(io.BytesIO(output.data)).size
        if profile is None or size != (profile.width, profile.height):
            raise RuntimeError(f"Qwen-Image CUDA smoke image is {size}, not the profile shape")
        print("[difflet serve] Qwen-Image CUDA generation smoke passed")

    def reset_request_state(self, outcome: str) -> None:
        return None

    async def shutdown(self) -> None:
        self.pipe = None
        self.active_profile = None
        self.active_runtime = None


__all__: list[str] = [
    "NOOP_GUIDANCE_SCALE",
    "SMOKE_STEPS",
    "CudaQwenImagePipelineRunner",
    "CudaQwenImageServingArtifactPreparer",
    "CudaQwenImageServingRequestValidator",
    "CudaQwenImageServingStageAdapter",
    "cuda_runtime_plan",
    "refuse_unsupported_profile",
]
