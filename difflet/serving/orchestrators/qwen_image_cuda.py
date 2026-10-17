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

Why the runtime plan claims core ``0``: the resident worker validates its allocation generically
(one allocation, positive core count, enough ``available_core_ids``) and exports
``NEURON_RT_VISIBLE_CORES`` from it. Those "cores" are abstract device slots to the engine; the
CUDA worker never loads the Neuron runtime, so the one GPU is slot 0 and the variable is inert.
"""

from __future__ import annotations

from typing import Any

from difflet.serving.errors import feature_not_supported
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators.base import resolve_hf_model_source
from difflet.serving.types import (
    ArtifactSet,
    CompileArtifactIdentity,
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


def _refuse(what: str) -> None:
    raise feature_not_supported(
        f"{_TAG} {what} is not supported on the CUDA backend yet"
    )


def refuse_unsupported_profile(profile: ServingProfile) -> None:
    """Raise ``feature_not_supported`` for every profile the CUDA backend does not serve."""
    if profile.shapes:
        _refuse("--shapes (bucketed request shapes)")
    parallel = profile.parallel
    if int(parallel.tp_degree) > 1:
        _refuse("--tp-degree > 1 (tensor parallel across PCIe)")
    if parallel.sp_enabled:
        _refuse("--sp (sequence parallel needs tp > 1)")
    if int(parallel.cp_degree) > 1:
        _refuse("--cp-degree > 1 (context parallel, second card)")
    if parallel.cfg_parallel_enabled:
        _refuse("--cfg-parallel (second card)")
    if profile.teacache_speedup is not None:
        _refuse("--teacache-speedup (TeaCache port)")


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


__all__: list[str] = [
    "CudaQwenImageServingArtifactPreparer",
    "cuda_runtime_plan",
    "refuse_unsupported_profile",
]
