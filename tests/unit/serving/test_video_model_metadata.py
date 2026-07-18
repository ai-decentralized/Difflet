from __future__ import annotations

import pytest

from difflet.common.registry import hunyuan_video, ltx_2, wan


@pytest.mark.parametrize(
    ("factory", "model_type", "steps", "guidance", "fps", "host_vae"),
    [
        (ltx_2.serving_metadata, "ltx_2", 40, 3.5, 24, True),
        (wan.serving_metadata, "wan", 2, 1.0, 16, True),
        (hunyuan_video.serving_metadata, "hunyuan_video", 4, 6.0, 24, True),
    ],
)
def test_video_metadata_uses_cli_defaults_and_mp4_contract(
    factory, model_type, steps, guidance, fps, host_vae
):
    metadata = factory()

    assert metadata.model_type == model_type
    assert metadata.output_modality == "video"
    assert metadata.output_mime_type == "video/mp4"
    assert metadata.chat_content_type == "video_url"
    assert metadata.default_steps == steps
    assert metadata.default_guidance_scale == guidance
    assert metadata.default_fps == fps
    assert metadata.default_host_vae is host_vae
    assert metadata.pipeline_definition.model_type == model_type


def test_ltx_2_uses_one_opaque_hybrid_pipeline():
    metadata = ltx_2.serving_metadata()

    assert metadata.checkpoint_ids == ("Lightricks/LTX-2",)
    assert [
        (
            stage.stage_id,
            stage.kind,
            stage.role,
            stage.output_keys,
            stage.final_output,
            stage.runner_factory,
        )
        for stage in metadata.pipeline_definition.stages
    ] == [("pipeline", "opaque_pipeline", "pipeline", ("output",), True, None)]


def test_wan_keeps_21_and_experimental_22_on_the_current_closed_loop_topology():
    metadata = wan.serving_metadata()

    assert metadata.checkpoint_ids == (
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    assert [
        (stage.stage_id, stage.role, stage.output_keys, stage.runner_factory)
        for stage in metadata.pipeline_definition.stages
    ] == [
        (
            "prompt_encoder",
            "prompt_encoder",
            ("prompt_embeds", "negative_prompt_embeds"),
            "difflet.serving.models.wan:WanPromptEncoderStageRunner",
        ),
        (
            "denoiser",
            "denoiser",
            ("latents",),
            "difflet.serving.models.wan:WanDenoiserStageRunner",
        ),
        (
            "decoder",
            "decoder",
            ("output",),
            "difflet.serving.models.wan:WanHostDecoderStageRunner",
        ),
    ]
    assert metadata.pipeline_definition.stages[-1].final_output is True


def test_hunyuan_video_uses_host_clip_and_decode_around_neuron_stages():
    metadata = hunyuan_video.serving_metadata()

    assert metadata.checkpoint_ids == ("hunyuanvideo-community/HunyuanVideo",)
    assert [
        (stage.stage_id, stage.role, stage.output_keys, stage.runner_factory)
        for stage in metadata.pipeline_definition.stages
    ] == [
        (
            "clip",
            "prompt_encoder",
            ("pooled_projections",),
            "difflet.serving.models.hunyuan_video:HunyuanVideoHostClipStageRunner",
        ),
        (
            "llama",
            "prompt_encoder",
            ("encoder_hidden_states", "encoder_attention_mask"),
            "difflet.serving.models.hunyuan_video:HunyuanVideoLlamaStageRunner",
        ),
        (
            "denoiser",
            "denoiser",
            ("latents",),
            "difflet.serving.models.hunyuan_video:HunyuanVideoDenoiserStageRunner",
        ),
        (
            "decoder",
            "decoder",
            ("output",),
            "difflet.serving.models.hunyuan_video:HunyuanVideoHostDecoderStageRunner",
        ),
    ]
    assert metadata.pipeline_definition.stages[-1].final_output is True


def test_hunyuan_15_remains_unregistered_while_wan_22_is_explicitly_experimental():
    assert "Wan-AI/Wan2.2-T2V-A14B-Diffusers" in wan.serving_metadata().checkpoint_ids
    hunyuan_ids = hunyuan_video.serving_metadata().checkpoint_ids
    assert "tencent/HunyuanVideo" not in hunyuan_ids
    assert not any("1.5" in checkpoint for checkpoint in hunyuan_ids)
