"""Factory for the P0 serving stack."""

from __future__ import annotations

from dataclasses import dataclass

from difflet.serving.engines import ResidentWorkerServingEngine
from difflet.serving.engines.resident_worker import ResidentWorkerConfig
from difflet.serving.model_registry import (
    ResolvedServingModel,
    load_preflight_factory,
    load_request_validator_factory,
    resolve_serving_model,
)
from difflet.serving.options import ServeOptions
from difflet.serving.orchestrators.base import NoopServingRequestValidator


@dataclass(frozen=True)
class ServingStack:
    resolved_model: ResolvedServingModel
    engine: ResidentWorkerServingEngine
    request_validator: object


def build_serving_stack(options: ServeOptions) -> ServingStack:
    resolved = resolve_serving_model(options)
    preflight_cls = load_preflight_factory(resolved.metadata)
    preflight = preflight_cls(model_id=resolved.model_id, revision=options.revision)
    preflight.resolve_model_path(download_policy=options.download_policy)
    print("[difflet serve] stage topology:")
    for stage in preflight.stage_specs(resolved.profile):
        print(f"  - {stage.stage_id}: role={stage.role} cores={stage.num_cores}")
    preflight.ensure_artifacts(resolved.profile, options.compile_policy)
    validator_factory = load_request_validator_factory(resolved.metadata)
    request_validator = (
        validator_factory(model_id=resolved.model_id, revision=options.revision)
        if validator_factory is not None
        else NoopServingRequestValidator()
    )
    engine = ResidentWorkerServingEngine(
        profile=resolved.profile,
        orchestrator_factory=resolved.metadata.orchestrator_factory,
        config=ResidentWorkerConfig(
            max_running_requests=options.max_running_requests,
            max_queued_requests=options.max_queued_requests,
            queue_timeout=options.queue_timeout,
            request_timeout=options.request_timeout,
            worker_cancel_timeout=options.worker_cancel_timeout,
            worker_restart_timeout=options.worker_restart_timeout,
        ),
    )
    return ServingStack(
        resolved_model=resolved,
        engine=engine,
        request_validator=request_validator,
    )
