"""FLUX.1-dev calibration loop for ``scripts/ptq_calibrate_activations.py`` (CPU, real weights).

Loaded by the driver for ``--model-type flux``; it calls ``run(args, install_hooks)``.

The transformer is the stock diffusers ``FluxTransformer2DModel`` so module names are
exactly the HF checkpoint keys (``transformer_blocks.3.attn.to_q``,
``single_transformer_blocks.7.proj_out`` -- the fused single-block projection, whose input
absmax is what ``difflet.quant.checkpoint.calibrated_amax`` looks up). The loop is the stock
diffusers ``FluxPipeline`` (what ``NeuronFluxPipeline`` defers to without true CFG): same
sigmas / shift ``mu`` / ``FlowMatchEulerDiscreteScheduler``, guidance-distilled so one
transformer pass per step, latents from ``torch.Generator().manual_seed(seed)`` like the CLI.

Conditioning: CLIP-L pooled + T5-XXL (``args.text_seq_len``, 512 on device) computed here
from ``args.prompt``; or, with ``--text-pt``, a ``torch.save``'d dict
``{"prompt_embeds": [1, seq, 4096], "pooled_prompt_embeds": [1, 768]}``.
The text encoders are freed before the transformer is loaded.

    DIFFLET_BACKEND=cpu PYTHONPATH=$PWD python scripts/ptq_calibrate_activations.py \\
        --model-type flux --model-dir <FLUX.1-dev snapshot> --height 1024 --width 1024 \\
        --steps 28 --guidance-scale 3.5 --out artifacts/.../flux_act_calibration.json
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

DTYPE = torch.bfloat16
# Per double block: q/k/v/out + add_q/k/v/to_add_out + ff in/out + ff_context in/out.
# Per single block: q/k/v + proj_mlp + fused proj_out.
LINEARS_PER_DOUBLE = 12
LINEARS_PER_SINGLE = 5


def expected_hooks(config) -> int:
    return LINEARS_PER_DOUBLE * config.num_layers + LINEARS_PER_SINGLE * config.num_single_layers


def encode_text(model_dir: Path, prompt: str, seq_len: int) -> dict[str, torch.Tensor]:
    """Real CLIP pooled + T5 embeddings via the stock ``FluxPipeline.encode_prompt``."""
    from diffusers import FluxPipeline
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    pipe = FluxPipeline(
        scheduler=None, vae=None, transformer=None,
        text_encoder=CLIPTextModel.from_pretrained(str(model_dir / "text_encoder"), torch_dtype=DTYPE).eval(),
        tokenizer=CLIPTokenizer.from_pretrained(str(model_dir / "tokenizer")),
        text_encoder_2=T5EncoderModel.from_pretrained(str(model_dir / "text_encoder_2"), torch_dtype=DTYPE).eval(),
        tokenizer_2=T5TokenizerFast.from_pretrained(str(model_dir / "tokenizer_2")),
    )
    with torch.no_grad():
        prompt_embeds, pooled, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=torch.device("cpu"),
                                                      max_sequence_length=seq_len)
    del pipe
    gc.collect()
    return {"prompt_embeds": prompt_embeds.to(DTYPE), "pooled_prompt_embeds": pooled.to(DTYPE)}


def load_transformer(model_dir: Path):
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel.from_pretrained(str(model_dir / "transformer"), torch_dtype=DTYPE).eval()


def load_scheduler(model_dir: Path):
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(str(model_dir / "scheduler"))


def run(args, install_hooks) -> float:
    from diffusers import FluxPipeline

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    if args.text_pt:
        text = torch.load(args.text_pt, map_location="cpu")
    else:
        text = encode_text(args.model_dir, args.prompt, args.text_seq_len)
    prompt_embeds = text["prompt_embeds"].to(DTYPE)
    pooled = text["pooled_prompt_embeds"].to(DTYPE)
    print(f"[calib] flux conditioning t5 {tuple(prompt_embeds.shape)} clip {tuple(pooled.shape)} "
          f"in {time.perf_counter() - started:.1f}s", flush=True)

    started = time.perf_counter()
    model = load_transformer(args.model_dir)
    print(f"[calib] transformer loaded in {time.perf_counter() - started:.1f}s", flush=True)
    hooked = install_hooks(model)
    want = expected_hooks(model.config)
    assert hooked == want, (f"hooked {hooked} linears, expected {want} for {model.config.num_layers} double + "
                            f"{model.config.num_single_layers} single blocks")

    # A CPU loop: keep the stock pipeline from calling xm.mark_step() when torch_xla is importable.
    import diffusers.pipelines.flux.pipeline_flux as pipeline_flux

    pipeline_flux.XLA_AVAILABLE = False
    pipe = FluxPipeline(scheduler=load_scheduler(args.model_dir), vae=None, text_encoder=None, tokenizer=None,
                        text_encoder_2=None, tokenizer_2=None, transformer=model)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            pipe(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled, height=args.height, width=args.width,
                 num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                 generator=torch.Generator().manual_seed(args.seed), max_sequence_length=prompt_embeds.shape[1],
                 output_type="latent")
    except Exception as exc:  # the driver's --max-steps _Stop: keep the real elapsed time
        if type(exc).__name__ != "_Stop":
            raise
    return time.perf_counter() - started
