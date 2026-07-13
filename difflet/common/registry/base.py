"""Shared serving model metadata."""

from __future__ import annotations

from dataclasses import dataclass

from difflet.serving.types import PipelineDefinition


@dataclass(frozen=True)
class ServingModelMetadata:
    model_type: str
    checkpoint_ids: tuple[str, ...]
    output_modality: str
    output_mime_type: str
    chat_content_type: str
    default_steps: int
    default_guidance_scale: float
    artifact_preparer_factory: str
    orchestrator_factory: str
    request_validator_factory: str | None = None
    pipeline_definition: PipelineDefinition | None = None

    def __post_init__(self) -> None:
        if self.pipeline_definition is None:
            raise ValueError("serving metadata requires pipeline_definition")
        if self.pipeline_definition.model_type != self.model_type:
            raise ValueError("pipeline model_type must match serving metadata")
