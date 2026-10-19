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


def cuda_serving_metadata() -> ServingModelMetadata:
    """FLUX.1-dev served on the CUDA backend (plan F3c): one resident process, one stage.

    Same checkpoint, same chat surface and defaults as the CUDA CLI (28 steps, distilled guidance
    3.5); the three factories live in ``difflet.serving.orchestrators.flux_cuda``.
    """
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
            "difflet.serving.orchestrators.flux_cuda:CudaFluxServingArtifactPreparer"
        ),
        orchestrator_factory="difflet.serving.orchestrators.flux_cuda:CudaFluxServingStageAdapter",
        request_validator_factory=(
            "difflet.serving.orchestrators.flux_cuda:CudaFluxServingRequestValidator"
        ),
    )
