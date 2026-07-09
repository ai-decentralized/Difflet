"""Flux common/serving metadata."""

from __future__ import annotations

from difflet.common.registry.base import ServingModelMetadata, ServingStageMetadata


def serving_metadata() -> ServingModelMetadata:
    return ServingModelMetadata(
        model_type="flux",
        checkpoint_ids=("black-forest-labs/FLUX.1-dev",),
        output_modality="image",
        output_mime_type="image/png",
        chat_content_type="image_url",
        default_steps=28,
        default_guidance_scale=3.5,
        stages=(
            ServingStageMetadata(
                stage_id="pipeline",
                role="pipeline",
                output_keys=("image",),
                final_output=True,
            ),
        ),
        preflight_factory="difflet.serving.orchestrators.flux:FluxServingArtifactPreparer",
        orchestrator_factory="difflet.serving.orchestrators.flux:FluxServingOrchestrator",
        request_validator_factory="difflet.serving.orchestrators.flux:FluxServingRequestValidator",
    )
