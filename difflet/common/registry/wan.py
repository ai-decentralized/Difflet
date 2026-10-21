"""Wan 2.1 common/serving metadata."""

from __future__ import annotations

from difflet.common.registry.base import ServingModelMetadata
from difflet.serving.types import PipelineDefinition, StageDefinition

_RUNNER_MODULE = "difflet.serving.models.wan"


def serving_metadata() -> ServingModelMetadata:
    """Describe the closed-loop Wan 2.1 resident serving topology.

    Prompt encoding and denoising use the existing Neuron components. The fixed
    serving profile defaults to the accepted Neuron VAE decoder; ``--host-vae``
    preserves the CLI-compatible host rollback and remains required for longer
    clips whose single-shot Neuron graph exceeds the compiler instruction
    limit. Wan 2.2 is accepted only as an explicit experimental checkpoint using
    this current single-transformer topology; it is not MVP-qualified because
    its second transformer is not loaded.
    """

    return ServingModelMetadata(
        model_type="wan",
        checkpoint_ids=(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        ),
        output_modality="video",
        output_mime_type="video/mp4",
        chat_content_type="video_url",
        default_steps=2,
        default_guidance_scale=1.0,
        pipeline_definition=PipelineDefinition(
            model_type="wan",
            stages=(
                StageDefinition(
                    stage_id="prompt_encoder",
                    kind="extracted",
                    role="prompt_encoder",
                    output_keys=("prompt_embeds", "negative_prompt_embeds"),
                    runner_factory=f"{_RUNNER_MODULE}:WanPromptEncoderStageRunner",
                ),
                StageDefinition(
                    stage_id="denoiser",
                    kind="extracted",
                    role="denoiser",
                    output_keys=("latents",),
                    runner_factory=f"{_RUNNER_MODULE}:WanDenoiserStageRunner",
                ),
                StageDefinition(
                    stage_id="decoder",
                    kind="extracted",
                    role="decoder",
                    output_keys=("output",),
                    final_output=True,
                    runner_factory=f"{_RUNNER_MODULE}:WanHostDecoderStageRunner",
                ),
            ),
        ),
        artifact_preparer_factory=f"{_RUNNER_MODULE}:WanServingArtifactPreparer",
        orchestrator_factory=f"{_RUNNER_MODULE}:WanServingStageAdapter",
        request_validator_factory=f"{_RUNNER_MODULE}:WanServingRequestValidator",
        default_fps=16,
        default_host_vae=False,
    )
