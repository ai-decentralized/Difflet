"""MiniMax-H3 common/serving metadata."""

from __future__ import annotations

from difflet.common.registry.base import ServingModelMetadata
from difflet.serving.types import PipelineDefinition, StageDefinition


def serving_metadata() -> ServingModelMetadata:
    """Describe MiniMax-H3's resident text-to-video(+audio) pipeline on the neuron backend.

    One opaque stage: the conditioner (loaded per request, released before the denoise), the
    resident DiT and the TP-sharded video decoder run on every rank of the worker's mesh, the
    audio decode and the MP4 mux on the primary replica. H3 is guidance-distilled: there is no
    unconditional pass, so the guidance scale is accepted and ignored.
    """

    return ServingModelMetadata(
        model_type="minimax_h3",
        checkpoint_ids=("MiniMaxAI/MiniMax-H3",),
        output_modality="video",
        output_mime_type="video/mp4",
        chat_content_type="video_url",
        default_steps=30,
        default_guidance_scale=1.0,
        pipeline_definition=PipelineDefinition(
            model_type="minimax_h3",
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
            "difflet.serving.models.minimax_h3:MiniMaxH3ServingArtifactPreparer"
        ),
        orchestrator_factory="difflet.serving.models.minimax_h3:MiniMaxH3ServingStageAdapter",
        request_validator_factory=(
            "difflet.serving.models.minimax_h3:MiniMaxH3ServingRequestValidator"
        ),
        default_fps=24,
        default_host_vae=False,
        output_has_audio=True,
    )
