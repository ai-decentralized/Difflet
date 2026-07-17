"""LTX-2 common/serving metadata."""

from __future__ import annotations

from difflet.common.registry.base import ServingModelMetadata
from difflet.serving.types import PipelineDefinition, StageDefinition


def serving_metadata() -> ServingModelMetadata:
    """Describe the existing hybrid host/Neuron LTX-2 pipeline.

    LTX-2 already owns prompt encoding and decode on the host around its
    Neuron-resident DiT.  Keep that boundary opaque until the lower-level
    pipeline exposes independently reusable stage contracts.
    """

    return ServingModelMetadata(
        model_type="ltx_2",
        checkpoint_ids=("Lightricks/LTX-2",),
        output_modality="video",
        output_mime_type="video/mp4",
        chat_content_type="video_url",
        default_steps=40,
        default_guidance_scale=3.5,
        pipeline_definition=PipelineDefinition(
            model_type="ltx_2",
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
        artifact_preparer_factory=("difflet.serving.models.ltx_2:LTX2ServingArtifactPreparer"),
        orchestrator_factory="difflet.serving.models.ltx_2:LTX2ServingStageAdapter",
        request_validator_factory=("difflet.serving.models.ltx_2:LTX2ServingRequestValidator"),
        default_fps=24,
        default_host_vae=True,
    )
