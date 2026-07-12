from __future__ import annotations

import pytest

from difflet.serving.types import (
    CompileArtifactIdentity,
    DistributedProcessEnvironment,
    PipelineDefinition,
    RuntimeEnvironment,
    RuntimePlan,
    StageDefinition,
    StageRuntimeSpec,
    ParallelTopology,
    WorkerAllocationSpec,
)


def _pipeline() -> PipelineDefinition:
    return PipelineDefinition(
        model_type="qwen_image",
        stages=(
            StageDefinition(
                stage_id="text",
                kind="extracted",
                role="prompt_encoder",
                runner_factory="pkg:TextRunner",
            ),
            StageDefinition(
                stage_id="vae",
                kind="extracted",
                role="decoder",
                final_output=True,
                runner_factory="pkg:VaeRunner",
            ),
        ),
    )


def test_pipeline_requires_unique_ids_and_last_final_stage():
    stage = StageDefinition(
        stage_id="pipeline",
        kind="opaque_pipeline",
        role="pipeline",
        final_output=True,
    )

    with pytest.raises(ValueError, match="unique"):
        PipelineDefinition(model_type="flux", stages=(stage, stage))

    with pytest.raises(ValueError, match="final stage"):
        PipelineDefinition(
            model_type="flux",
            stages=(
                stage,
                StageDefinition(
                    stage_id="other",
                    kind="opaque_pipeline",
                    role="pipeline",
                ),
            ),
        )


def test_stage_kind_controls_runner_factory():
    with pytest.raises(ValueError, match="requires runner_factory"):
        StageDefinition(stage_id="text", kind="extracted", role="prompt_encoder")
    with pytest.raises(ValueError, match="must not define runner_factory"):
        StageDefinition(
            stage_id="pipeline",
            kind="opaque_pipeline",
            role="pipeline",
            runner_factory="pkg:Runner",
        )


def test_runtime_plan_validates_pipeline_order_and_allocation():
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment(
        available_core_ids=(0, 1, 2, 3),
        num_cores_override=None,
        virtual_core_size_override=None,
        logical_nc_config_override=None,
        inherited_distributed=distributed,
        child_distributed=distributed,
    )
    allocation = WorkerAllocationSpec("worker", 4, 4, 4)
    topology = ParallelTopology(tp_degree=4, cp_degree=1, world_size=4)
    plan = RuntimePlan(
        mode="resident",
        profile_identity="profile",
        environment=environment,
        allocations=(allocation,),
        stages=(
            StageRuntimeSpec("text", "worker", topology, "text-artifact"),
            StageRuntimeSpec("vae", "worker", topology, "vae-artifact"),
        ),
    )

    plan.validate_against(_pipeline())

    reversed_plan = RuntimePlan(
        mode=plan.mode,
        profile_identity=plan.profile_identity,
        environment=plan.environment,
        allocations=plan.allocations,
        stages=tuple(reversed(plan.stages)),
    )
    with pytest.raises(ValueError, match="do not match pipeline"):
        reversed_plan.validate_against(_pipeline())


def test_compile_identity_is_canonical_and_rejects_nonfinite_numbers():
    first = CompileArtifactIdentity.from_cache_inputs({"b": 2, "a": [1, True]})
    second = CompileArtifactIdentity.from_cache_inputs({"a": [1, True], "b": 2})

    assert first == second
    with pytest.raises(ValueError):
        CompileArtifactIdentity.from_cache_inputs({"bad": float("nan")})
