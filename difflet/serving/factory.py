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
from difflet.serving.orchestrators.base import NoopServingRequestValidator
from difflet.serving.types import ResolvedRuntimeBundle


@dataclass(frozen=True)
class ServingStack:
    resolved_model: ResolvedServingModel
    runtime: ResolvedRuntimeBundle
    engine: ResidentWorkerServingEngine
    request_validator: object


def build_serving_stack(options: ServeOptions) -> ServingStack:
    resolved = resolve_serving_model(options)
    preparer_cls = load_artifact_preparer_factory(resolved.metadata)
    preparer = preparer_cls(model_id=resolved.model_id, revision=options.revision)
    runtime = preparer.prepare_runtime(
        resolved.profile,
        download_policy=options.download_policy,
        compile_policy=options.compile_policy,
    )
    print("[difflet serve] stage topology:")
    for stage in runtime.pipeline_definition.stages:
        print(f"  - {stage.stage_id}: role={stage.role} kind={stage.kind}")
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
            max_queued_requests=options.max_queued_requests,
            queue_timeout=options.queue_timeout,
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
