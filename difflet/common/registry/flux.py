"""Flux common/serving metadata."""

from __future__ import annotations

from difflet.common.registry.base import ServingModelMetadata
from difflet.serving.types import PipelineDefinition, StageDefinition


def serving_metadata() -> ServingModelMetadata:
    return ServingModelMetadata(
        model_type="flux",
        checkpoint_ids=("black-forest-labs/FLUX.1-dev",),
        output_modality="image",
        output_mime_type="image/png",
        chat_content_type="image_url",
        default_steps=28,
        default_guidance_scale=3.5,
        pipeline_definition=PipelineDefinition(
            model_type="flux",
            stages=(
                StageDefinition(
                    stage_id="pipeline",
                    kind="opaque_pipeline",
                    role="pipeline",
                    output_keys=("output",),
                    final_output=True,
                ),
            ),
        ),
        artifact_preparer_factory=(
            "difflet.serving.orchestrators.flux:FluxServingArtifactPreparer"
        ),
        orchestrator_factory="difflet.serving.orchestrators.flux:FluxServingStageAdapter",
        request_validator_factory="difflet.serving.orchestrators.flux:FluxServingRequestValidator",
    )
