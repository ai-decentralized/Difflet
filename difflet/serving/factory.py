"""Factory for the P0 serving stack."""

from __future__ import annotations

from dataclasses import dataclass

from difflet.serving.engines import ResidentWorkerServingEngine
from difflet.serving.engines.resident_worker import ResidentWorkerConfig
from difflet.serving.model_registry import (
    ResolvedServingModel,
    load_artifact_preparer_factory,
    load_request_validator_factory,
    resolve_serving_model,
)
from difflet.serving.options import ServeOptions
from difflet.serving.orchestrators.base import (
    NoopServingRequestValidator,
    ServingArtifactPreparer,
    ServingRequestValidator,
)
from difflet.serving.types import ResolvedRuntimeBundle


@dataclass(frozen=True)
class ServingStack:
    resolved_model: ResolvedServingModel
    runtime: ResolvedRuntimeBundle
    engine: ResidentWorkerServingEngine
    request_validator: ServingRequestValidator


def build_serving_stack(options: ServeOptions) -> ServingStack:
    resolved = resolve_serving_model(options)
    preparer_cls = load_artifact_preparer_factory(resolved.metadata)
    preparer: ServingArtifactPreparer = preparer_cls(
        model_id=resolved.model_id,
        revision=options.revision,
    )
    runtime = preparer.prepare_runtime(
        resolved.profile,
        download_policy=options.download_policy,
        compile_policy=options.compile_policy,
    )
    print("[difflet serve] stage topology:")
    for stage, stage_runtime in zip(
        runtime.pipeline_definition.stages,
        runtime.runtime_plan.stages,
        strict=True,
    ):
        print(
            f"  - {stage.stage_id}: role={stage.role} kind={stage.kind} "
            f"placement={stage_runtime.placement} artifact={stage_runtime.artifact_id or '-'}"
        )
    validator_factory = load_request_validator_factory(resolved.metadata)
    request_validator = (
        validator_factory(runtime)
        if validator_factory is not None
        else NoopServingRequestValidator(runtime)
    )
    engine = ResidentWorkerServingEngine(
        runtime=runtime,
        orchestrator_factory=resolved.metadata.orchestrator_factory,
        config=ResidentWorkerConfig(
            max_running_requests=options.max_running_requests,
            # Video sync and async work share VideoGenerationService's FIFO.
            # Keeping the inner engine queue disabled prevents either API path
            # from bypassing that single admission domain.
            max_queued_requests=(
                0 if resolved.metadata.output_modality == "video" else options.max_queued_requests
            ),
            queue_timeout=options.effective_queue_timeout(
                resolved.metadata.output_modality
            ),
            request_timeout=options.request_timeout,
            worker_cancel_timeout=options.worker_cancel_timeout,
            worker_restart_timeout=options.worker_restart_timeout,
            worker_heartbeat_interval=options.worker_heartbeat_interval,
        ),
    )
    return ServingStack(
        resolved_model=resolved,
        runtime=runtime,
        engine=engine,
        request_validator=request_validator,
    )
