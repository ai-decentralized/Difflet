"""LTX-2.0 audiovisual inference on the generic GPU flow."""

from difflet.models.spec import ModelSpec, PromptBudget

TEXT_SEQ_LEN = 1024

SPEC = ModelSpec(
    model_type="ltx_2",
    display_name="LTX-2",
    short_name="ltx2",
    cli_model_ids=("Lightricks/LTX-2",),
    served_checkpoint_ids=("Lightricks/LTX-2",),
    output="video",
    default_height=512,
    default_width=768,
    default_num_frames=121,
    default_steps=40,
    guidance="cfg",
    default_guidance_scale=4.0,
    default_fps=24,
    accepts_negative_prompt=True,
    refused_cli_flags=(
        ("guidance_scale_2", "--guidance-scale-2 (LTX-2 has one joint transformer)"),
        ("true_cfg_scale", "--true-cfg-scale (use --guidance-scale for LTX-2 true CFG)"),
        ("dual_expert", "--dual-expert (LTX-2 has one joint transformer)"),
        ("device_vae", "--device-vae (Neuron decoder placement)"),
        ("vae_in_process", "--vae-in-process (Neuron decoder placement)"),
        ("vae_dtype", "--vae-dtype (separate decoder precision)"),
        ("vae_exec_mode", "--vae-exec-mode (Neuron decoder execution)"),
        ("host_text_encoder", "--host-text-encoder (host text encoder placement)"),
        ("text_encoder_dtype", "--text-encoder-dtype (separate text encoder precision)"),
        ("device_text_encoder", "--device-text-encoder (Neuron text encoder placement)"),
        ("exec_mode", "--exec-mode (use --torch-compile for the GPU DiT)"),
        ("tp_reduce_dtype", "--tp-reduce-dtype (tensor parallelism)"),
        ("shapes", "--shapes (bucketed request shapes)"),
        ("teacache_cadence", "--teacache-cadence (TeaCache for LTX-2)"),
        ("teacache_speedup", "--teacache-speedup (TeaCache for LTX-2)"),
        ("teacache_online_delta", "--teacache-online-delta (TeaCache for LTX-2)"),
        ("teacache_calibration", "--teacache-calibration (TeaCache for LTX-2)"),
        ("taef1", "--taef1 / --taef1-path (tiny VAE decoder)"),
        ("taef1_path", "--taef1 / --taef1-path (tiny VAE decoder)"),
        ("host_vae", "--host-vae (the VAE already runs in-process on the GPU)"),
    ),
    refused_serving_options=(
        ("teacache_speedup", "--teacache-speedup (TeaCache for LTX-2)"),
        ("host_vae", "--host-vae (the VAE already runs on the GPU)"),
    ),
    refused_video_fields=(
        ("guidance_scale_2", "LTX-2 uses one joint transformer; use guidance_scale"),
        ("boundary_ratio", "LTX-2 uses one joint transformer"),
        ("flow_shift", "the scheduler shift is the checkpoint's"),
        ("true_cfg_scale", "LTX-2 already uses true CFG; use guidance_scale"),
    ),
    refused_application_kwargs=("shapes", "taef1", "taef1_path", "host_vae", "sp_enabled"),
    refused_application_kwarg_prefixes=("teacache_",),
    refused_application_kwargs_reason="one resident shape; no TeaCache, TAEF1, host VAE or SP",
    prompt_budgets=(
        PromptBudget(
            tokenizer_subdir="tokenizer",
            max_tokens=TEXT_SEQ_LEN,
            too_long="LTX-2 {name} exceeds Gemma's 1024 tokens",
            applies_to_negative_prompt=True,
        ),
    ),
    application="difflet.backends.cuda.ltx_2.application:CudaLtx2Application",
)
