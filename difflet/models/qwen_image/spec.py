"""Qwen-Image on the generic GPU flow."""

from difflet.models.spec import ModelSpec, PromptBudget

#: The DiT's text length; the encoder truncates past it silently, so serving refuses longer
#: prompts. ``difflet.common.orchestrators.qwen_image.TEXT_SEQ_LEN``, kept torch-free here.
TEXT_SEQ_LEN = 1024

#: diffusers' QwenImagePipeline prompt template and the prefix length it drops.
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_DROP_TOKENS = 34

SPEC = ModelSpec(
    model_type="qwen_image",
    display_name="Qwen-Image",
    short_name="qwen",
    cli_model_ids=("Qwen/Qwen-Image",),
    served_checkpoint_ids=("Qwen/Qwen-Image",),
    output="image",
    default_height=1024,
    default_width=1024,
    default_num_frames=None,
    default_steps=50,  # diffusers' QwenImagePipeline default
    guidance="none",
    default_guidance_scale=None,
    serving_noop_guidance_scale=4.0,
    no_guidance_reason=(
        "the transformer has guidance_embeds=False and generation is one pass per step "
        "without classifier-free guidance"
    ),
    refused_cli_flags=(
        ("shapes", "--shapes (bucketed request shapes)"),
        ("teacache_speedup", "--teacache-speedup (TeaCache probe path)"),
        ("teacache_online_delta", "--teacache-online-delta (TeaCache probe path)"),
    ),
    # The probe-free fixed TeaCache cadence runs on the host-side controller.
    forwarded_cli_flags=("torch_compile", "teacache_cadence"),
    refused_serving_options=(("teacache_speedup", "--teacache-speedup (TeaCache port)"),),
    refused_application_kwargs=(
        "teacache_speedup",
        "teacache_calibration",
        "teacache_calibration_path",
        "teacache_online_delta_alpha",
    ),
    refused_application_kwargs_reason=("use teacache_cadence (probe-free) or run without TeaCache"),
    prompt_budgets=(
        PromptBudget(
            tokenizer_subdir="tokenizer",
            max_tokens=TEXT_SEQ_LEN,
            too_long=f"Qwen prompt exceeds the DiT text length {TEXT_SEQ_LEN}",
            template=PROMPT_TEMPLATE,
            drop_prefix_tokens=PROMPT_DROP_TOKENS,
        ),
    ),
    application="difflet.backends.cuda.qwen_image.application:CudaQwenImageApplication",
)
