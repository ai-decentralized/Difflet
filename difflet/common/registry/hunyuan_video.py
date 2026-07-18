"""HunyuanVideo 1.0 common/serving metadata."""

from __future__ import annotations

from difflet.common.registry.base import ServingModelMetadata
from difflet.serving.types import PipelineDefinition, StageDefinition

_RUNNER_MODULE = "difflet.serving.models.hunyuan_video"


def serving_metadata() -> ServingModelMetadata:
    """Describe the honest HunyuanVideo 1.0 resident stage boundaries.

    The validated default keeps CLIP and VAE execution on the host. The Llama
    encoder and denoiser use their existing Neuron artifacts. Runtime profile
    resolution may replace the CLIP and decoder bindings with explicit
    experimental Neuron placements. HunyuanVideo 1.5 remains unregistered until
    its offline compile and generation path is complete.
    """

    return ServingModelMetadata(
        model_type="hunyuan_video",
        checkpoint_ids=("hunyuanvideo-community/HunyuanVideo",),
        output_modality="video",
        output_mime_type="video/mp4",
        chat_content_type="video_url",
        default_steps=4,
        default_guidance_scale=6.0,
        pipeline_definition=PipelineDefinition(
            model_type="hunyuan_video",
            stages=(
                StageDefinition(
                    stage_id="clip",
                    kind="extracted",
                    role="prompt_encoder",
                    output_keys=("pooled_projections",),
                    runner_factory=f"{_RUNNER_MODULE}:HunyuanVideoHostClipStageRunner",
                ),
                StageDefinition(
                    stage_id="llama",
                    kind="extracted",
                    role="prompt_encoder",
                    output_keys=("encoder_hidden_states", "encoder_attention_mask"),
                    runner_factory=f"{_RUNNER_MODULE}:HunyuanVideoLlamaStageRunner",
                ),
                StageDefinition(
                    stage_id="denoiser",
                    kind="extracted",
                    role="denoiser",
                    output_keys=("latents",),
                    runner_factory=f"{_RUNNER_MODULE}:HunyuanVideoDenoiserStageRunner",
                ),
                StageDefinition(
                    stage_id="decoder",
                    kind="extracted",
                    role="decoder",
                    output_keys=("output",),
                    final_output=True,
                    runner_factory=f"{_RUNNER_MODULE}:HunyuanVideoHostDecoderStageRunner",
                ),
            ),
        ),
        artifact_preparer_factory=(f"{_RUNNER_MODULE}:HunyuanVideoServingArtifactPreparer"),
        orchestrator_factory=f"{_RUNNER_MODULE}:HunyuanVideoServingStageAdapter",
        request_validator_factory=(f"{_RUNNER_MODULE}:HunyuanVideoServingRequestValidator"),
        default_fps=24,
        default_host_vae=True,
    )
