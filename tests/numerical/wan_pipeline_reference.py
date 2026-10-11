"""CPU reference for the end-to-end Wan2.1 check (W4): diffusers ``WanPipeline``, latents out.

The oracle is the upstream pipeline, not another Difflet backend: tokenizer, umT5 encode of the
prompt and of the empty negative prompt, UniPC (flow_shift 3.0), classifier-free guidance with
two batch-1 transformer calls per step, all as diffusers runs them. The init noise is
``torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=float32)`` and is
passed as ``latents=``, so a comparison does not depend on generator call order. The VAE is not
run (``output_type="latent"``).

Three runs share the noise and the fp32 prompt embeddings:

* ``fp32``: the reference (transformer and umT5 in fp32);
* ``bf16``: the control, the same pipeline with the transformer cast to bf16 in place (what
  bf16 alone costs upstream; the fp32 embeddings are cast to bf16 by the pipeline, as the
  device receives the fp32 umT5 output broadcast in bf16);
* ``bf16_tsround`` (diagnostic): the control with every timestep rounded to bf16 before the
  transformer sees it (999 -> 1000, 750 -> 752), which is what Difflet's orchestrator feeds a
  bf16 DiT (``difflet/models/wan/pipeline.py`` ``_batch_timestep``, decision D44), so a
  device-vs-reference gap from that rounding can be told apart from the DiT's own.

Saved with ``torch.save`` (rewritten after each run, so a late failure keeps the earlier ones):
``meta`` (prompt, negative prompt, steps, guidance, seed, shape, scheduler timesteps, versions,
snapshot, seconds), ``latents_init``, ``prompt_embeds`` / ``negative_prompt_embeds`` (fp32
``(1, max_sequence_length, text_dim)``, zero after the valid tokens), ``prompt_tokens`` /
``negative_prompt_tokens`` (valid token counts), and ``runs[name]`` = ``latents_steps`` (the
latents after each scheduler step), ``noise_preds`` (every transformer output in call order,
conditional then unconditional per step, as fp32), ``timesteps_seen`` (what the transformer
received), ``latents_final`` and ``seconds``.

Memory: the 14B fp32 transformer (57 GB) plus umT5 fp32 (22.7 GB) while encoding; umT5 is
dropped after the encode. Run it alone, never next to a device launch:

    HF_HUB_OFFLINE=1 PYTHONSAFEPATH=1 PYTHONPATH=<worktree> python \\
        <worktree>/tests/numerical/wan_pipeline_reference.py --model-dir <snapshot> \\
        --prompt "a cinematic shot of a red fox running through a snowy forest" --steps 2 \\
        --guidance-scale 5.0 --seed 42 --threads 12 --out <records>/ref_2steps.pt
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Any

import torch

RUNS = ("fp32", "bf16", "bf16_tsround")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True, help="diffusers Wan snapshot")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="",
                   help="diffusers' default (an empty prompt), also Difflet's orchestrator's")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--runs", default=",".join(RUNS),
                   help=f"comma-separated subset of {RUNS}, in that order; fp32 is required")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.runs = [r for r in args.runs.split(",") if r]
    unknown = sorted(set(args.runs) - set(RUNS))
    if unknown or "fp32" not in args.runs:
        p.error(f"--runs must name fp32 and only {RUNS}, got {args.runs}")
    args.runs = [r for r in RUNS if r in args.runs]
    if not args.model_dir.is_dir():
        p.error(f"--model-dir {args.model_dir} is not a directory")
    if not args.out.parent.is_dir():
        p.error(f"the directory of --out {args.out} does not exist")
    return args


def init_latents(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    """The shared init noise: a fresh CPU generator, fp32 (also what the device check uses)."""
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)


def latent_shape(pipe: Any, args: argparse.Namespace) -> tuple[int, ...]:
    channels = int(pipe.transformer.config.in_channels)
    frames = (args.num_frames - 1) // int(pipe.vae_scale_factor_temporal) + 1
    spatial = int(pipe.vae_scale_factor_spatial)
    return (1, channels, frames, args.height // spatial, args.width // spatial)


def valid_tokens(pipe: Any, prompt: str, max_length: int) -> int:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    mask = pipe.tokenizer([prompt_clean(prompt)], padding="max_length", max_length=max_length,
                          truncation=True, add_special_tokens=True,
                          return_attention_mask=True, return_tensors="pt").attention_mask
    return int(mask.sum())


def vm_hwm_gib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def run_pipeline(pipe: Any, args: argparse.Namespace, *, latents: torch.Tensor,
                 prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor | None,
                 round_timestep: bool) -> dict[str, Any]:
    """One ``WanPipeline`` call on the given noise and embeddings, recording every step."""
    record: dict[str, Any] = {"latents_steps": [], "noise_preds": [], "timesteps_seen": []}

    def before(module, call_args, kwargs):
        timestep = kwargs["timestep"]
        if round_timestep:
            timestep = timestep.to(torch.bfloat16).to(torch.float32)
            kwargs = {**kwargs, "timestep": timestep}
        record["timesteps_seen"].append(float(timestep.flatten()[0]))
        return call_args, kwargs

    def after(module, call_args, kwargs, output):
        record["noise_preds"].append(output[0].detach().to(torch.float32).clone())

    def on_step_end(pipeline, index, timestep, callback_kwargs):
        record["latents_steps"].append(callback_kwargs["latents"].detach().clone())
        return {}

    hooks = [pipe.transformer.register_forward_pre_hook(before, with_kwargs=True),
             pipe.transformer.register_forward_hook(after, with_kwargs=True)]
    start = time.perf_counter()
    try:
        with torch.no_grad():
            out = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                latents=latents.clone(),
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                output_type="latent",
                callback_on_step_end=on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                max_sequence_length=args.max_sequence_length,
            )
    finally:
        for hook in hooks:
            hook.remove()
    record["seconds"] = time.perf_counter() - start
    record["latents_final"] = out.frames.detach().to(torch.float32).clone()
    expected = args.steps * (2 if args.guidance_scale > 1.0 else 1)
    if len(record["noise_preds"]) != expected or len(record["latents_steps"]) != args.steps:
        raise RuntimeError(f"recorded {len(record['noise_preds'])} transformer calls and "
                           f"{len(record['latents_steps'])} steps, expected {expected} and "
                           f"{args.steps}")
    return record


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    import diffusers
    import transformers
    from diffusers import WanPipeline

    t0 = time.perf_counter()
    print(f"loading WanPipeline fp32 from {args.model_dir} ({args.threads} threads) ...",
          flush=True)
    pipe = WanPipeline.from_pretrained(str(args.model_dir), torch_dtype=torch.float32,
                                       local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    load_s = time.perf_counter() - t0
    shape = latent_shape(pipe, args)
    latents = init_latents(args.seed, shape)
    print(f"  loaded in {load_s:.1f}s; latents {shape}; VmHWM {vm_hwm_gib():.1f} GiB",
          flush=True)

    cfg = args.guidance_scale > 1.0
    start = time.perf_counter()
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            do_classifier_free_guidance=cfg, max_sequence_length=args.max_sequence_length,
            device=torch.device("cpu"), dtype=torch.float32)
    encode_s = time.perf_counter() - start
    tokens = valid_tokens(pipe, args.prompt, args.max_sequence_length)
    negative_tokens = valid_tokens(pipe, args.negative_prompt, args.max_sequence_length)
    print(f"  encoded prompt ({tokens} tokens) and negative prompt ({negative_tokens}) in "
          f"{encode_s:.1f}s", flush=True)
    # umT5 is not needed any more: drop its 22.7 GB before the transformer runs.
    pipe.text_encoder = None
    gc.collect()

    pipe.scheduler.set_timesteps(args.steps)
    timesteps = [int(t) for t in pipe.scheduler.timesteps]
    blob: dict[str, Any] = {
        "meta": {
            "prompt": args.prompt, "negative_prompt": args.negative_prompt,
            "steps": args.steps, "guidance_scale": args.guidance_scale, "seed": args.seed,
            "height": args.height, "width": args.width, "num_frames": args.num_frames,
            "max_sequence_length": args.max_sequence_length, "latent_shape": list(shape),
            "timesteps": timesteps, "snapshot": str(args.model_dir.resolve()),
            "threads": args.threads, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "runs": list(args.runs),
            "seconds": {"load": load_s, "encode": encode_s},
        },
        "latents_init": latents,
        "prompt_embeds": prompt_embeds.detach().to(torch.float32),
        "negative_prompt_embeds": (None if negative_prompt_embeds is None
                                   else negative_prompt_embeds.detach().to(torch.float32)),
        "prompt_tokens": tokens,
        "negative_prompt_tokens": negative_tokens,
        "runs": {},
    }
    for name in args.runs:
        if name != "fp32" and pipe.transformer.dtype != torch.bfloat16:
            start = time.perf_counter()
            pipe.transformer.to(torch.bfloat16)
            gc.collect()
            print(f"  transformer cast to bf16 in {time.perf_counter() - start:.1f}s",
                  flush=True)
        record = run_pipeline(pipe, args, latents=latents, prompt_embeds=prompt_embeds,
                              negative_prompt_embeds=negative_prompt_embeds,
                              round_timestep=name == "bf16_tsround")
        blob["runs"][name] = record
        blob["meta"]["seconds"][name] = record["seconds"]
        blob["meta"]["vm_hwm_gib"] = vm_hwm_gib()
        torch.save(blob, args.out)
        print(f"  {name}: {args.steps} steps in {record['seconds']:.1f}s, timesteps seen "
              f"{record['timesteps_seen']}, final latents absmean "
              f"{float(record['latents_final'].abs().mean()):.4f}; saved {args.out}; "
              f"VmHWM {blob['meta']['vm_hwm_gib']:.1f} GiB", flush=True)
    print(f"done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
