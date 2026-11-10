"""HunyuanVideo 1.5 text-to-video on the GPU flow."""

from difflet.models.spec import ModelSpec

MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

SPEC = ModelSpec(
    model_type="hunyuan_video_15",
    display_name="HunyuanVideo 1.5",
    short_name="hunyuan15",
    cli_model_ids=(MODEL_ID,),
    served_checkpoint_ids=(MODEL_ID,),
    output="video",
    default_height=480,
    default_width=848,
    default_num_frames=121,
    default_steps=50,
    default_fps=24,
    guidance="cfg",
    default_guidance_scale=6.0,
    accepts_negative_prompt=True,
    accepts_cfg_parallel=False,
    refused_cli_flags=(
        ("shapes", "--shapes"),
        ("teacache_cadence", "--teacache-cadence"),
        ("teacache_speedup", "--teacache-speedup"),
        ("teacache_online_delta", "--teacache-online-delta"),
        ("teacache_calibration", "--teacache-calibration"),
        ("host_vae", "--host-vae"),
        ("taef1", "--taef1"),
        ("taef1_path", "--taef1-path"),
        ("true_cfg_scale", "--true-cfg-scale"),
        ("guidance_scale_2", "--guidance-scale-2"),
    ),
    refused_serving_options=(
        ("teacache_speedup", "--teacache-speedup"),
        ("host_vae", "--host-vae"),
    ),
    refused_video_fields=(
        ("true_cfg_scale", "use guidance_scale for standard CFG"),
        ("guidance_scale_2", "there is one transformer"),
        ("boundary_ratio", "there is one transformer"),
        ("flow_shift", "the scheduler configuration comes from the checkpoint"),
    ),
    refused_application_kwargs=("shapes", "host_vae", "taef1", "taef1_path"),
    refused_application_kwarg_prefixes=("teacache_",),
    refused_application_kwargs_reason="one resident shape with standard CFG",
    # No prompt budgets: Diffusers' driver crops its own chat template and glyph text.
    smoke_prompt="a cat walking across a sunlit garden",
    application="difflet.backends.cuda.hunyuan_video_15.application:CudaHunyuanVideo15Application",
)
