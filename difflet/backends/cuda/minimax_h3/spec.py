"""MiniMax-H3 on CUDA, through the generic GPU flow: text-to-video and first-frame
image-to-video, each with a stereo audio track.

This spec is the CUDA backend's alone. The neuron backend declares the model separately: its
serving metadata in ``difflet/common/registry/minimax_h3.py`` (keyframes as one flag, first
and last), its defaults in ``difflet/common/orchestrators/minimax_h3.py``.
"""

from difflet.models.spec import ModelSpec, PromptBudget

_TEACACHE = "(TeaCache is not implemented for MiniMax-H3)"
_DISTILLED = "the released checkpoint is guidance-distilled"

SPEC = ModelSpec(
    model_type="minimax_h3",
    display_name="MiniMax-H3",
    short_name="minimax-h3",
    cli_model_ids=("MiniMaxAI/MiniMax-H3",),
    served_checkpoint_ids=("MiniMaxAI/MiniMax-H3",),
    output="video",
    # The registry's shape: the checkpoint's native 1344x768, 5 s at 24 fps.
    default_height=768,
    default_width=1344,
    default_num_frames=124,
    default_steps=50,
    guidance="none",
    default_guidance_scale=None,
    # One pass per step: a served request may only carry CFG's no-op scale.
    serving_noop_guidance_scale=1.0,
    no_guidance_reason=_DISTILLED,
    # First-frame conditioning (I2VA) only.
    keyframe_anchors=("first",),
    default_fps=24,
    output_has_audio=True,
    refused_cli_flags=(
        ("shapes", "--shapes (bucketed request shapes)"),
        ("teacache_cadence", f"--teacache-cadence {_TEACACHE}"),
        ("teacache_speedup", f"--teacache-speedup {_TEACACHE}"),
        ("teacache_online_delta", f"--teacache-online-delta {_TEACACHE}"),
        ("teacache_calibration", f"--teacache-calibration {_TEACACHE}"),
        ("taef1", "--taef1 / --taef1-path (tiny VAE decoder)"),
        ("taef1_path", "--taef1 / --taef1-path (tiny VAE decoder)"),
        ("host_vae", "--host-vae (the VAEs already run in-process on the GPU)"),
        ("guidance_scale_2", f"--guidance-scale-2 ({_DISTILLED})"),
        ("true_cfg_scale", f"--true-cfg-scale ({_DISTILLED})"),
    ),
    refused_serving_options=(
        ("teacache_speedup", f"--teacache-speedup {_TEACACHE}"),
        ("host_vae", "--host-vae (the VAEs already run on the GPU)"),
    ),
    refused_video_fields=(
        ("negative_prompt", _DISTILLED),
        ("guidance_scale_2", _DISTILLED),
        ("true_cfg_scale", _DISTILLED),
        ("boundary_ratio", "MiniMax-H3 has one transformer"),
        ("flow_shift", "the scheduler shifts are the checkpoint's"),
    ),
    refused_application_kwargs=("shapes",),
    refused_application_kwarg_prefixes=("teacache_",),
    refused_application_kwargs_reason="one request shape on one GPU; no TeaCache",
    prompt_budgets=(
        PromptBudget(
            tokenizer_subdir="tokenizer",
            max_tokens=1024,
            too_long="MiniMax-H3 {name} exceeds 1024 text tokens",
        ),
    ),
    application="difflet.backends.cuda.minimax_h3.application:CudaMiniMaxH3Application",
    # The files the CUDA application loads: the repository's root diffusers partition, without
    # the Ref2VA transformer and the original FL2VA/Ref2VA checkpoints beside it.
    download_patterns=(
        "*.json",
        "*.txt",
        "*.md",
        "LICENSE",
        "transformer/diffusion_pytorch_model*.safetensors",
        "text_encoder/model*.safetensors",
        "tokenizer/*",
        "processor/*",
        "vae/diffusion_pytorch_model*.safetensors",
        "audio_vae/diffusion_pytorch_model*.safetensors",
    ),
)
