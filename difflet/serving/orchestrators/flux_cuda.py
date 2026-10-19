"""FLUX.1-dev served on the CUDA backend — one resident process, one stage.

``qwen_image_cuda``'s shape, applied to Flux: at ``tp=1`` the whole model is
one ``CudaFluxApplication`` behind ``DiffletPipeline``, so the resident worker gets a single
opaque ``pipeline`` stage that holds ``DiffletPipeline.from_pretrained(backend="cuda")`` and calls
``app.generate()`` per request — the two calls ``difflet run`` makes on CUDA
(``difflet/cli/orchestrators/flux_cuda.py``), with the same kwargs. Selected by backend name in
``difflet.serving.model_registry``; the engine, the HTTP layer and the Trainium Flux adapter
(``flux.py``) are untouched. ``--torch-compile`` reaches the application as it does on the CLI.

The request contract is the Trainium Flux validator's: the resident profile's shape, a distilled
``guidance_scale`` in [0, 20] (one pass per step, FLUX.1-dev's default 3.5), and a prompt within
T5's 512 tokens (``max_sequence_length``; ``FluxPipeline`` would truncate it silently). Release 1
scope: tp=1, cp=1, one request shape, no TeaCache — anything wider is refused in
``prepare_runtime``, before the 34 GB snapshot is resolved.
"""

from __future__ import annotations

import io
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from difflet.common.orchestrators import flux as flux_common
from difflet.serving.engines.stage_pipeline import (
    ErasedStageRunner,
    ValidatedStageRunner,
    require_exact_payload,
    stage_result,
)
from difflet.serving.errors import feature_not_supported, profile_mismatch, prompt_too_long
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators import cuda_resident
from difflet.serving.orchestrators.base import resolve_hf_model_source, validate_guidance_scale
from difflet.serving.types import (
    ArtifactSet,
    DiffletGenerateOutput,
    DiffletGenerateRequest,
    FluxFinalPayload,
    FluxInitialPayload,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    RuntimePlan,
    ServingProfile,
    StageExecutionResult,
    StageInvocation,
    StagePayload,
)

_HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"
_MODEL_TYPE = "flux"
_ALLOCATION_ID = "flux-cuda"
_STAGE_ID = cuda_resident.STAGE_ID
_TAG = "[flux/cuda serve]"
#: FLUX.1-dev's distilled guidance default (FluxPipeline's), the startup smoke's value.
DEFAULT_GUIDANCE_SCALE = 3.5
MAX_GUIDANCE_SCALE = 20.0  # the Trainium Flux validator's bound
MAX_SEQUENCE_LENGTH = flux_common.MAX_SEQUENCE_LENGTH  # T5 tokens, 512
#: Startup smoke: cheap, like Qwen's on CUDA; proves load + one generate.
SMOKE_STEPS = 4


def refuse_unsupported_profile(profile: ServingProfile) -> None:
    """Raise ``feature_not_supported`` for every profile the CUDA backend does not serve."""
    cuda_resident.refuse_shared_profile(profile, tag=_TAG)
    if profile.parallel.cfg_parallel_enabled:
        # One transformer pass per step, no second branch to place.
        raise feature_not_supported(
            f"{_TAG} --cfg-parallel does not apply to FLUX.1-dev: it is guidance-distilled "
            "(the guidance scale is an input of one pass per step, no conditional/"
            "unconditional pair)"
        )
    if profile.teacache_speedup is not None:
        cuda_resident.refuse(_TAG, "--teacache-speedup (TeaCache for Flux)")


def cuda_runtime_plan(profile: ServingProfile, source: ResolvedModelSource) -> RuntimePlan:
    """The resident plan for one GPU: one allocation (slot 0), one eager ``cuda`` stage."""
    return cuda_resident.resident_runtime_plan(
        profile, source, allocation_id=_ALLOCATION_ID, stage_id=_STAGE_ID
    )


class CudaFluxServingArtifactPreparer:
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
        print(f"[difflet serve] resolving FLUX.1-dev weights for {self.model_id}")
        source = resolve_hf_model_source(
            self.model_id, revision=self.revision, download_policy=download_policy
        )
        # No compile phase (eager): DiffletPipeline writes its manifest-only
        # artifact inside the worker at load, as for Qwen-Image.
        print("[difflet serve] cuda backend: eager resident pipeline, no compile artifacts")
        from difflet.common.registry.flux import cuda_serving_metadata

        return ResolvedRuntimeBundle(
            profile=profile,
            source=source,
            pipeline_definition=cuda_serving_metadata().pipeline_definition,
            runtime_plan=cuda_runtime_plan(profile, source),
            compile_specs=(),
            artifacts=ArtifactSet(()),
            adapter_config={"compile_policy": compile_policy},
        )


class CudaFluxServingRequestValidator:
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
        validate_guidance_scale(request, maximum=MAX_GUIDANCE_SCALE)
        profile = self.runtime.profile
        if (request.height, request.width) != (profile.height, profile.width):
            raise profile_mismatch(
                f"request shape {request.height}x{request.width} is not the resident CUDA "
                f"profile shape {profile.height}x{profile.width}"
            )
        encoded = self._tokenizer_for_runtime()(
            request.prompt, padding=False, truncation=False, return_tensors="pt"
        )
        if len(encoded.input_ids[0]) > MAX_SEQUENCE_LENGTH:
            raise prompt_too_long(f"Flux prompt exceeds T5's {MAX_SEQUENCE_LENGTH} tokens")

    def _tokenizer_for_runtime(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_dir = self.runtime.source.pinned_model_path
            self._tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer_2"))
        return self._tokenizer


class CudaFluxPipelineRunner:
    """The one stage: prompt in, PNG out, through the resident ``DiffletPipeline``."""

    def __init__(self, adapter: "CudaFluxServingStageAdapter") -> None:
        self.adapter = adapter

    async def execute(
        self,
        invocation: StageInvocation[FluxInitialPayload],
    ) -> StageExecutionResult[FluxFinalPayload]:
        started = time.monotonic()
        invocation.context.cancellation.throw_if_cancelled()
        output = self.adapter.generate_png(invocation.request)
        return stage_result(FluxFinalPayload(output=output), started_monotonic=started)

    async def shutdown(self) -> None:
        self.adapter.pipe = None


class CudaFluxServingStageAdapter:
    """``ServingStageAdapter`` for FLUX.1-dev on CUDA: the model stays resident in the worker."""

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
        print("[difflet serve] loading FLUX.1-dev on CUDA (CLIP, T5, DiT, VAE in-process)")
        self.active_runtime = runtime
        self.active_profile = profile
        # The CLI's kwargs (FluxCudaOrchestrator.pipeline_kwargs); the revision is the commit
        # the parent pinned, so a worker restart loads the snapshot the bundle describes.
        self.pipe = DiffletPipeline.from_pretrained(
            self.model_id,
            **cuda_pipeline_kwargs(
                tp_degree=profile.parallel.tp_degree,
                height=profile.height,
                width=profile.width,
                cache_dir=profile.cache_dir,
                revision=runtime.source.resolved_source_id,
                force_compile=config.get("compile_policy") is CompilePolicy.FORCE,
                load=True,
                torch_compile=profile.torch_compile,
            ),
        )
        print("[difflet serve] FLUX.1-dev CUDA pipeline resident")
        return OrderedDict(
            (
                (
                    _STAGE_ID,
                    ValidatedStageRunner(
                        CudaFluxPipelineRunner(self), FluxInitialPayload, FluxFinalPayload
                    ),
                ),
            )
        )

    def generate_png(self, request: DiffletGenerateRequest) -> DiffletGenerateOutput:
        """``app.generate`` with the CLI's arguments (``FluxCudaOrchestrator.generate_kwargs``)."""
        if self.pipe is None:
            raise RuntimeError("FLUX.1-dev CUDA pipeline is not loaded")
        image = self.pipe.app.generate(
            request.prompt,
            num_inference_steps=int(request.num_inference_steps),
            guidance_scale=float(request.guidance_scale),
            seed=int(request.seed),
            output_type="pil",
        )[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return DiffletGenerateOutput(
            data=buf.getvalue(), mime_type="image/png", output_format="png"
        )

    def initial_payload(self, request: DiffletGenerateRequest) -> FluxInitialPayload:
        return FluxInitialPayload()

    def finalize(self, payload: StagePayload) -> DiffletGenerateOutput:
        return require_exact_payload(
            payload, FluxFinalPayload, boundary="Flux CUDA final payload"
        ).output

    def smoke_request(self) -> DiffletGenerateRequest:
        if self.pipe is None or self.active_profile is None:
            raise RuntimeError("FLUX.1-dev CUDA load did not leave a resident pipeline")
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
        )

    def validate_smoke_output(self, output: DiffletGenerateOutput) -> None:
        if not output.data:
            raise RuntimeError("FLUX.1-dev CUDA smoke produced empty output")
        from PIL import Image

        profile = self.active_profile
        size = Image.open(io.BytesIO(output.data)).size
        if profile is None or size != (profile.width, profile.height):
            raise RuntimeError(f"FLUX.1-dev CUDA smoke image is {size}, not the profile shape")
        print("[difflet serve] FLUX.1-dev CUDA generation smoke passed")

    def reset_request_state(self, outcome: str) -> None:
        return None

    async def shutdown(self) -> None:
        self.pipe = None
        self.active_profile = None
        self.active_runtime = None


__all__: list[str] = [
    "DEFAULT_GUIDANCE_SCALE",
    "MAX_GUIDANCE_SCALE",
    "MAX_SEQUENCE_LENGTH",
    "SMOKE_STEPS",
    "CudaFluxPipelineRunner",
    "CudaFluxServingArtifactPreparer",
    "CudaFluxServingRequestValidator",
    "CudaFluxServingStageAdapter",
    "cuda_runtime_plan",
    "refuse_unsupported_profile",
]
