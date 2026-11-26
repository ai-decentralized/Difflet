"""Wan 2.2 T2V-A14B and Wan 2.1 T2V-14B on the generic GPU flow.

One spec, one set of defaults (Wan 2.2's: 40 steps, guidance 4.0). Wan 2.1's model card runs
50 steps at guidance 5.0; pass ``--steps 50 --guidance-scale 5.0`` for it.
"""

from difflet.models.spec import ModelSpec, PromptBudget

WAN22 = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WAN21 = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

TEXT_SEQ_LEN = 512  # UMT5 tokens (WanPipeline's max_sequence_length)

_TEACACHE = "(TeaCache for Wan)"

SPEC = ModelSpec(
    model_type="wan",
    display_name="Wan",
    short_name="wan",
    cli_model_ids=(WAN22, WAN21),
    served_checkpoint_ids=(WAN22, WAN21),
    output="video",
    default_height=480,
    default_width=832,
    default_num_frames=9,
    default_steps=40,  # WanPipeline's Wan 2.2 default
    guidance="cfg",
    default_guidance_scale=4.0,  # every step on Wan 2.1, the high-noise stage on Wan 2.2
    default_fps=16,
    # Wan 2.1 has one transformer and no expert boundary.
    guidance_scale_2_checkpoint_ids=(WAN22,),
    accepts_cfg_parallel=True,
    accepts_negative_prompt=True,
    refused_cli_flags=(
        ("shapes", "--shapes (bucketed request shapes)"),
        ("teacache_cadence", f"--teacache-cadence {_TEACACHE}"),
        ("teacache_speedup", f"--teacache-speedup {_TEACACHE}"),
        ("teacache_online_delta", f"--teacache-online-delta {_TEACACHE}"),
        ("teacache_calibration", f"--teacache-calibration {_TEACACHE}"),
        ("taef1", "--taef1 / --taef1-path (tiny VAE decoder)"),
        ("taef1_path", "--taef1 / --taef1-path (tiny VAE decoder)"),
        ("host_vae", "--host-vae (the VAE already runs in-process on the GPU)"),
    ),
    refused_serving_options=(
        ("teacache_speedup", f"--teacache-speedup {_TEACACHE}"),
        ("host_vae", "--host-vae (the VAE already runs on the GPU)"),
    ),
    refused_video_fields=(
        ("boundary_ratio", "the expert boundary is the checkpoint's (model_index.json)"),
        ("flow_shift", "the scheduler shift is the checkpoint's"),
        (
            "true_cfg_scale",
            "Wan already runs true CFG; use guidance_scale (and guidance_scale_2 on Wan 2.2)",
        ),
    ),
    refused_application_kwargs=("shapes",),
    refused_application_kwarg_prefixes=("teacache_",),
    refused_application_kwargs_reason="one request shape at tp=1; no TeaCache",
    prompt_budgets=(
        PromptBudget(
            tokenizer_subdir="tokenizer",
            max_tokens=TEXT_SEQ_LEN,
            too_long="Wan {name} exceeds UMT5's 512 tokens",
            applies_to_negative_prompt=True,
        ),
    ),
    application="difflet.backends.cuda.wan.application:CudaWanApplication",
)
