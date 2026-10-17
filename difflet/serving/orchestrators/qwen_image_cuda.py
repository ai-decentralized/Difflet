"""Qwen-Image served on the CUDA backend — one resident process, one stage (CUDA plan F3b, D5).

The Trainium adapter (``qwen_image.py``) fills the serving engine's resident worker with three
Neuron stage applications and a NeuronCore allocation. On CUDA at ``tp=1`` the whole model is one
``CudaQwenImageApplication`` behind ``DiffletPipeline``, so the worker gets a single opaque
``pipeline`` stage that holds ``DiffletPipeline.from_pretrained(backend="cuda")`` resident and
calls ``app.generate()`` per request — the same two calls ``difflet run`` makes on CUDA
(``difflet/cli/orchestrators/qwen_image_cuda.py``). Selected by backend name in
``difflet.serving.model_registry`` (``cuda_serving_metadata``); the engine, the HTTP layer and
the Trainium adapter are untouched.

Release 1 scope: tp=1, cp=1, dp=1, no CFG-parallel, no TeaCache, one request shape. Anything
wider is refused in ``prepare_runtime`` — before the 54 GB snapshot is resolved — naming the
plan follow-up that owns it.

Why the runtime plan claims core ``0``: the resident worker validates its allocation generically
(one allocation, positive core count, enough ``available_core_ids``) and exports
``NEURON_RT_VISIBLE_CORES`` from it. Those "cores" are abstract device slots to the engine; the
CUDA worker never loads the Neuron runtime, so the one GPU is slot 0 and the variable is inert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from difflet.common.orchestrators.qwen_image import TEXT_SEQ_LEN
from difflet.serving.errors import feature_not_supported, profile_mismatch, prompt_too_long
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators.base import resolve_hf_model_source
from difflet.serving.types import (
    ArtifactSet,
    CompileArtifactIdentity,
    DiffletGenerateRequest,
    DistributedProcessEnvironment,
    ParallelTopology,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    RuntimeEnvironment,
    RuntimePlan,
    ServingProfile,
    StageRuntimeSpec,
    WorkerAllocationSpec,
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


def _refuse(what: str, follow_up: str) -> None:
    raise feature_not_supported(
        f"{_TAG} {what} is not supported on the CUDA backend in Release 1 "
        f"(plan follow-up {follow_up})"
    )


def refuse_unsupported_profile(profile: ServingProfile) -> None:
    """Raise ``feature_not_supported`` naming the follow-up for every profile outside Release 1."""
    if profile.shapes:
        _refuse("--shapes (bucketed request shapes)", "F3")
    parallel = profile.parallel
    if int(parallel.tp_degree) > 1:
        _refuse("--tp-degree > 1 (tensor parallel across PCIe)", "F6")
    if parallel.sp_enabled:
        _refuse("--sp (sequence parallel needs tp > 1)", "F6")
    if int(parallel.cp_degree) > 1:
        _refuse("--cp-degree > 1 (context parallel, second card)", "F2")
    if parallel.cfg_parallel_enabled:
        _refuse("--cfg-parallel (second card)", "F2")
    if profile.teacache_speedup is not None:
        _refuse("--teacache-speedup (TeaCache port)", "F4")


def cuda_runtime_plan(profile: ServingProfile, source: ResolvedModelSource) -> RuntimePlan:
    """The resident plan for one GPU: one allocation (slot 0), one eager ``cuda`` stage."""
    identity = CompileArtifactIdentity.from_cache_inputs(
        {
            "backend": "cuda",
            "model_type": profile.model_type,
            "source_id": source.resolved_source_id,
            "height": int(profile.height),
            "width": int(profile.width),
            "tp_degree": int(profile.parallel.tp_degree),
            "dtype": profile.dtype,
        }
    )
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment(
        available_core_ids=(0,),
        num_cores_override=None,
        virtual_core_size_override=None,
        logical_nc_config_override=None,
        inherited_distributed=distributed,
        child_distributed=distributed,
    )
    allocation = WorkerAllocationSpec(
        allocation_id=_ALLOCATION_ID,
        requested_num_cores=1,
        effective_num_cores=1,
        world_size=1,
    )
    stage = StageRuntimeSpec(
        stage_id=_STAGE_ID,
        allocation_id=_ALLOCATION_ID,
        topology=ParallelTopology(tp_degree=1, cp_degree=1, world_size=1),
        artifact_id=None,
        placement="cuda",
    )
    return RuntimePlan(
        mode="resident",
        profile_identity=identity.digest,
        environment=environment,
        allocations=(allocation,),
        stages=(stage,),
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
        # No compile phase: the CUDA backend is eager (plan D4), and the manifest-only
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


__all__: list[str] = [
    "NOOP_GUIDANCE_SCALE",
    "CudaQwenImageServingArtifactPreparer",
    "CudaQwenImageServingRequestValidator",
    "cuda_runtime_plan",
    "refuse_unsupported_profile",
]
