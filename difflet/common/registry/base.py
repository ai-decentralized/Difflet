"""Shared serving model metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass

from difflet.serving.types import OutputModality, PipelineDefinition


@dataclass(frozen=True)
class ServingModelMetadata:
    model_type: str
    checkpoint_ids: tuple[str, ...]
    output_modality: OutputModality
    output_mime_type: str
    chat_content_type: str
    default_steps: int
    default_guidance_scale: float
    artifact_preparer_factory: str
    orchestrator_factory: str
    request_validator_factory: str | None = None
    pipeline_definition: PipelineDefinition | None = None
    default_fps: int | None = None
    default_host_vae: bool = False

    def __post_init__(self) -> None:
        if self.output_modality not in {"image", "video"}:
            raise ValueError(f"unsupported serving output modality {self.output_modality!r}")
        if self.default_steps <= 0:
            raise ValueError("serving metadata default_steps must be positive")
        if not math.isfinite(self.default_guidance_scale):
            raise ValueError("serving metadata default_guidance_scale must be finite")
        if self.pipeline_definition is None:
            raise ValueError("serving metadata requires pipeline_definition")
        if self.pipeline_definition.model_type != self.model_type:
            raise ValueError("pipeline model_type must match serving metadata")
        if self.output_modality == "video":
            if self.output_mime_type != "video/mp4":
                raise ValueError("video serving metadata must use video/mp4")
            if self.default_fps is None or self.default_fps <= 0:
                raise ValueError("video serving metadata requires a positive default_fps")
        elif self.default_fps is not None or self.default_host_vae:
            raise ValueError("image serving metadata cannot define video placement defaults")
