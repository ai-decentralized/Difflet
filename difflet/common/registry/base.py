"""Compatibility-preserving wrapper around `difflet.registry`."""

from __future__ import annotations

from dataclasses import dataclass

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.registry import ModelEntry, resolve_model


@dataclass(frozen=True)
class CommonModelDescriptor:
    model_id: str
    entry: ModelEntry
    default_parallel: DiffletParallelConfig
    default_shape: dict[str, int | None]


@dataclass(frozen=True)
class ServingStageMetadata:
    stage_id: str
    role: str
    output_keys: tuple[str, ...] = ()
    final_output: bool = False


@dataclass(frozen=True)
class ServingModelMetadata:
    model_type: str
    checkpoint_ids: tuple[str, ...]
    output_modality: str
    output_mime_type: str
    chat_content_type: str
    default_steps: int
    default_guidance_scale: float
    preflight_factory: str
    orchestrator_factory: str
    request_validator_factory: str | None = None
    stages: tuple[ServingStageMetadata, ...] = ()


def describe_model(model_id: str, *, model_type: str | None = None) -> CommonModelDescriptor:
    entry = resolve_model(model_id, model_type=model_type)
    return CommonModelDescriptor(
        model_id=model_id,
        entry=entry,
        default_parallel=entry.default_parallel,
        default_shape=dict(entry.default_shape),
    )
