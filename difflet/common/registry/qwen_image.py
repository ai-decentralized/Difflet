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
            "difflet.serving.orchestrators.qwen_image:" "QwenImageServingStageAdapter"
        ),
        request_validator_factory=(
            "difflet.serving.orchestrators.qwen_image:" "QwenImageServingRequestValidator"
        ),
    )


def cuda_serving_metadata() -> ServingModelMetadata:
    """Qwen-Image served on the CUDA backend (plan F3b): one resident process, one stage.

    Same checkpoint, same chat surface; the request defaults are the CUDA CLI's (50 steps,
    guidance 4.0 — the value the application fills into the ignored guidance tensor) and the
    three factories live in ``difflet.serving.orchestrators.qwen_image_cuda``.
    """
    return ServingModelMetadata(
        model_type="qwen_image",
        checkpoint_ids=("Qwen/Qwen-Image",),
        output_modality="image",
        output_mime_type="image/png",
        chat_content_type="image_url",
        default_steps=50,
        default_guidance_scale=4.0,
        pipeline_definition=PipelineDefinition(
            model_type="qwen_image",
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
            "difflet.serving.orchestrators.qwen_image_cuda:CudaQwenImageServingArtifactPreparer"
        ),
        orchestrator_factory=(
            "difflet.serving.orchestrators.qwen_image_cuda:CudaQwenImageServingStageAdapter"
        ),
        request_validator_factory=(
            "difflet.serving.orchestrators.qwen_image_cuda:CudaQwenImageServingRequestValidator"
        ),
    )
