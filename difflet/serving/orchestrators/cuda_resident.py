"""The CUDA serving adapters' shared half.

Every model served on the CUDA backend is one ``DiffletPipeline`` resident in one worker
process on one GPU, behind a single opaque ``pipeline`` stage. What does not depend on the
model lives here: the resident ``RuntimePlan`` and the profile refusals every CUDA model
shares. The model adapters (``qwen_image_cuda``, ``flux_cuda``, ``wan_cuda``) add their own
request contract and ``generate`` call.

Why the plan claims core ``0``: the resident worker validates its allocation generically (one
allocation, positive core count, enough ``available_core_ids``) and exports
``NEURON_RT_VISIBLE_CORES`` from it. Those "cores" are abstract device slots to the engine; the
CUDA worker never loads the Neuron runtime, so the one GPU is slot 0 and the variable is inert
(``--dp-degree N`` rewrites the slot per replica).
"""

from __future__ import annotations

from difflet.serving.errors import feature_not_supported
from difflet.serving.types import (
    CompileArtifactIdentity,
    DistributedProcessEnvironment,
    ParallelTopology,
    ResolvedModelSource,
    RuntimeEnvironment,
    RuntimePlan,
    ServingProfile,
    StageRuntimeSpec,
    WorkerAllocationSpec,
)

STAGE_ID = "pipeline"


def refuse(tag: str, what: str) -> None:
    """Raise ``feature_not_supported`` for ``what``."""
    raise feature_not_supported(f"{tag} {what} is not supported on the CUDA backend yet")


def refuse_shared_profile(profile: ServingProfile, *, tag: str) -> None:
    """The refusals every CUDA profile shares: one request shape, one card, no context or
    sequence parallel."""
    if profile.shapes:
        refuse(tag, "--shapes (bucketed request shapes)")
    parallel = profile.parallel
    if int(parallel.tp_degree) > 1:
        refuse(tag, "--tp-degree > 1 (tensor parallel across PCIe)")
    if parallel.sp_enabled:
        refuse(tag, "--sp (sequence parallel needs tp > 1)")
    if int(parallel.cp_degree) > 1:
        refuse(tag, "--cp-degree > 1 (context parallel across cards)")


def resident_runtime_plan(
    profile: ServingProfile,
    source: ResolvedModelSource,
    *,
    allocation_id: str,
    stage_id: str = STAGE_ID,
) -> RuntimePlan:
    """The resident plan for one GPU: one allocation (slot 0), one eager ``cuda`` stage.

    The profile identity hashes the backend, model, pinned source, shape, tp and dtype — and
    ``num_frames`` for a video profile only, so an image profile's identity is Qwen-Image's."""
    inputs = {
        "backend": "cuda",
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
    allocation = WorkerAllocationSpec(
        allocation_id=allocation_id,
        requested_num_cores=1,
        effective_num_cores=1,
        world_size=1,
    )
    stage = StageRuntimeSpec(
        stage_id=stage_id,
        allocation_id=allocation_id,
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


__all__ = ["STAGE_ID", "refuse", "refuse_shared_profile", "resident_runtime_plan"]
