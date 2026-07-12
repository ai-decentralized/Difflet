"""Qwen-Image common/serving metadata."""

from __future__ import annotations

from difflet.common.registry.base import ServingModelMetadata
from difflet.serving.types import PipelineDefinition, StageDefinition


def serving_metadata() -> ServingModelMetadata:
    return ServingModelMetadata(
        model_type="qwen_image",
        checkpoint_ids=("Qwen/Qwen-Image",),
        output_modality="image",
        output_mime_type="image/png",
        chat_content_type="image_url",
        default_steps=4,
        default_guidance_scale=4.0,
        pipeline_definition=PipelineDefinition(
            model_type="qwen_image",
            stages=(
                StageDefinition(
                    stage_id="text",
                    kind="extracted",
                    role="prompt_encoder",
                    output_keys=("encoder_hidden_states", "encoder_hidden_states_mask"),
                    runner_factory=("difflet.serving.orchestrators.qwen_image:QwenTextStageRunner"),
                ),
                StageDefinition(
                    stage_id="generate",
                    kind="extracted",
                    role="denoiser",
                    output_keys=("packed_latents",),
                    runner_factory=(
                        "difflet.serving.orchestrators.qwen_image:QwenGenerateStageRunner"
                    ),
                ),
                StageDefinition(
                    stage_id="vae",
                    kind="extracted",
                    role="decoder",
                    output_keys=("output",),
                    final_output=True,
                    runner_factory=("difflet.serving.orchestrators.qwen_image:QwenVaeStageRunner"),
                ),
            ),
        ),
        artifact_preparer_factory=(
            "difflet.serving.orchestrators.qwen_image:" "QwenImageServingArtifactPreparer"
        ),
        orchestrator_factory=(
            "difflet.serving.orchestrators.qwen_image:" "QwenImageServingOrchestrator"
        ),
        request_validator_factory=(
            "difflet.serving.orchestrators.qwen_image:" "QwenImageServingRequestValidator"
        ),
    )
