"""Flux on the generic GPU flow."""

from difflet.models.spec import ModelSpec, PromptBudget

MAX_SEQUENCE_LENGTH = 512  # T5 prompt tokens (FluxPipeline's max_sequence_length)

_TEACACHE = "(TeaCache for Flux)"

SPEC = ModelSpec(
    model_type="flux",
    display_name="Flux",
    short_name="flux",
    cli_model_ids=("black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"),
    served_checkpoint_ids=("black-forest-labs/FLUX.1-dev",),
    output="image",
    default_height=1024,
    default_width=1024,
    default_num_frames=None,
    default_steps=28,  # FluxPipeline's default
    guidance="distilled",
    default_guidance_scale=3.5,  # FluxPipeline's default
    refused_cli_flags=(
        ("shapes", "--shapes (bucketed request shapes)"),
        ("teacache_cadence", f"--teacache-cadence {_TEACACHE}"),
        ("teacache_speedup", f"--teacache-speedup {_TEACACHE}"),
        ("teacache_online_delta", f"--teacache-online-delta {_TEACACHE}"),
        # main._validate_teacache lets a lone calibration file through and nothing forwards it
        # for Flux, so it would be dropped silently.
        ("teacache_calibration", f"--teacache-calibration {_TEACACHE}"),
        ("taef1", "--taef1 / --taef1-path (tiny VAE decoder)"),
        ("taef1_path", "--taef1 / --taef1-path (tiny VAE decoder)"),
        ("host_vae", "--host-vae (the VAE already runs in-process on the GPU)"),
    ),
    refused_serving_options=(("teacache_speedup", f"--teacache-speedup {_TEACACHE}"),),
    refused_application_kwargs=("taef1", "taef1_path", "shapes", "sp_enabled"),
    refused_application_kwarg_prefixes=("teacache_",),
    refused_application_kwargs_reason="one request shape at tp=1; no TeaCache, TAEF1 or SP",
    prompt_budgets=(
        PromptBudget(
            tokenizer_subdir="tokenizer_2",
            max_tokens=MAX_SEQUENCE_LENGTH,
            too_long=f"Flux prompt exceeds T5's {MAX_SEQUENCE_LENGTH} tokens",
        ),
    ),
    application="difflet.backends.cuda.flux.application:CudaFluxApplication",
)
