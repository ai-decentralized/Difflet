"""Original guidance-distilled HunyuanVideo on the generic GPU flow."""

from difflet.models.hunyuan_video.prompt import LLAMA_CROP_START, LLAMA_TEMPLATE, TEXT_SEQ_LEN
from difflet.models.spec import ModelSpec, PromptBudget

SPEC = ModelSpec(
    model_type="hunyuan_video",
    display_name="HunyuanVideo",
    short_name="hunyuan",
    cli_model_ids=("hunyuanvideo-community/HunyuanVideo",),
    served_checkpoint_ids=("hunyuanvideo-community/HunyuanVideo",),
    output="video",
    default_height=320,
    default_width=512,
    default_num_frames=61,
    default_steps=50,
    default_fps=24,
    height_multiple=16,
    width_multiple=16,
    frame_stride=4,
    guidance="distilled",
    default_guidance_scale=6.0,
    accepts_negative_prompt=False,
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
    ),
    refused_serving_options=(
        ("teacache_speedup", "--teacache-speedup"),
        ("host_vae", "--host-vae"),
        ("clip_placement", "--clip-placement"),
    ),
    refused_video_fields=(
        ("negative_prompt", "only distilled guidance is supported"),
        ("true_cfg_scale", "only distilled guidance is supported"),
        ("guidance_scale_2", "there is one transformer"),
        ("boundary_ratio", "there is one transformer"),
        ("flow_shift", "the scheduler configuration comes from the checkpoint"),
    ),
    refused_application_kwargs=("shapes", "host_vae", "taef1", "taef1_path", "clip_placement"),
    refused_application_kwarg_prefixes=("teacache_",),
    refused_application_kwargs_reason="one resident shape with distilled guidance",
    prompt_budgets=(
        PromptBudget(
            tokenizer_subdir="tokenizer",
            max_tokens=TEXT_SEQ_LEN,
            too_long="HunyuanVideo {name} exceeds Llama's 256 retained tokens",
            template=LLAMA_TEMPLATE,
            drop_prefix_tokens=LLAMA_CROP_START,
        ),
    ),
    smoke_prompt="a cat walking across a sunlit garden",
    application="difflet.backends.cuda.hunyuan_video.application:CudaHunyuanVideoApplication",
)
