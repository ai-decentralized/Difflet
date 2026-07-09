"""Qwen-Image common/serving metadata."""

from __future__ import annotations

from difflet.common.registry.base import ServingModelMetadata, ServingStageMetadata


def serving_metadata() -> ServingModelMetadata:
    return ServingModelMetadata(
        model_type="qwen_image",
        checkpoint_ids=("Qwen/Qwen-Image",),
        output_modality="image",
        output_mime_type="image/png",
        chat_content_type="image_url",
        default_steps=4,
        default_guidance_scale=4.0,
        stages=(
            ServingStageMetadata(
                stage_id="text",
                role="prompt_encoder",
                output_keys=("encoder_hidden_states", "encoder_hidden_states_mask"),
            ),
            ServingStageMetadata(
                stage_id="generate",
                role="denoiser",
                output_keys=("latents",),
            ),
            ServingStageMetadata(
                stage_id="vae",
                role="decoder",
                output_keys=("image",),
                final_output=True,
            ),
        ),
        preflight_factory=(
            "difflet.serving.orchestrators.qwen_image:"
            "QwenImageServingArtifactPreparer"
        ),
        orchestrator_factory=(
            "difflet.serving.orchestrators.qwen_image:"
            "QwenImageServingOrchestrator"
        ),
        request_validator_factory=(
            "difflet.serving.orchestrators.qwen_image:"
            "QwenImageServingRequestValidator"
        ),
    )
