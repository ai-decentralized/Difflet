#!/usr/bin/env python3
"""HunyuanVideo-1.5's CPU fp32 reference (phase G): diffusers' HunyuanVideo15Pipeline with every
component in fp32 on the host CPU, at the settings and from the initial noise of the 9-frame
device runs ``score.py`` scores against it (hv15's bf16_9 and nod4's mx, the shipped MXFP8
default on the tree without D4).

Every constant is copied from the phase G research object for HunyuanVideo-1.5
(``/home/ubuntu/fp8-bench/phase_g_research.json``). What makes this the device's run rather than
diffusers' defaults:

- true CFG at the snapshot guider's 6.0 against the empty negative prompt the device encodes,
  both branches every step; the script refuses a guider that is not that;
- the noise: the device draws it in bf16 (its application dtype) on a CPU generator, which is
  diffusers' fp32 seed-42 draw rounded to bf16 (rel L2 1.7e-3 from the unrounded one), so the
  bf16 draw is upcast (exact), its md5 checked and passed as ``latents``;
- the text: diffusers' own ``encode_prompt`` for the prompt and for "", as the device's host
  encode stage ran it in bf16;
- height, width and frames are passed: diffusers' defaults are 121 frames and a 640-target size;
- the decode tiles as the device decoder does (128-pixel tiles, 8 latent units, diffusers' 25%
  overlap), not the research recipe's untiled decode: tile-128 is 3.2e-2 from untiled (the device
  decoder's docstring), a floor both device arms would carry that would blur their difference;
- ``latents.pt`` is the last step's latents before ``/ scaling_factor``, (1, 32, 3, 30, 53) fp32,
  the device's (bf16) layout; the clip is the pipeline processor's frames encoded by
  ``save_run`` at 24 fps, as the device's VAE stage does with export_to_video.

The device's own deviations (a bf16 DiT on bf16 latents with bf16-rounded timesteps, the bf16
text encode and VAE) are shared by both device arms: they are what the reference measures.

    PYTHONPATH=. python scripts/fp32_ref/hunyuan_video_15.py --out-dir DIR [--threads 96] \
        [--steps 50]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.fp32_ref._common import (DEVICE_FPS, device_noise, require_fp32_math, save_run,
                                      set_threads, tensor_md5)

# research device_settings, from the arms' launch scripts (hv15/run_f6a.sh's bf16_9,
# run_nod4.sh's mx) and the transformer stage's defaults for what they leave out.
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
#: refs/main, the only local snapshot: the runs passed no --revision (gena.log's weight store).
REVISION = "286be7ce72277246578a3e3cc2487e95ddae5bcf"
#: The stage's defaults (no --height / --width) and the arms' --num-frames 9.
HEIGHT, WIDTH, NUM_FRAMES = 480, 848, 9
STEPS, SEED = 50, 42
PROMPT = "a cat walking through a garden"
#: The stage never forwards a negative prompt; its encode stage encodes ``negative or ""``.
NEGATIVE_PROMPT = ""
#: The snapshot guider's ClassifierFreeGuidance scale (no --guidance-scale: load_guidance).
GUIDANCE_SCALE = 6.0
#: The VAE's spatial compression (``VAE_SPATIAL``): latents (B, 32, (F - 1) / 4 + 1, H / 16,
#: W / 16), research latents_layout.
VAE_SPATIAL = 16
LATENT_SHAPE = (1, 32, 3, 30, 53)
#: md5 of the device's bf16 seed-42 draw at LATENT_SHAPE (hashed as bf16; research noise: equal
#: to diffusers' prepare_latents in bf16 and to its fp32 draw rounded to bf16).
NOISE_MD5 = "aad07dc00a94c602772871a1a157c1b7"
#: NeuronHunyuanVideo15VaeDecoderApplication's default tile, in sample pixels.
VAE_TILE = 128
FPS = DEVICE_FPS["hunyuan_video_15"]


def initial_noise() -> torch.Tensor:
    """The device's initial latents as fp32: its bf16 seed-42 draw, upcast (exact), refused if
    its md5 is not the device draw's."""
    noise = device_noise(LATENT_SHAPE, SEED, bf16_draw=True)
    md5 = tensor_md5(noise.to(torch.bfloat16))
    if md5 != NOISE_MD5:
        raise RuntimeError(f"initial noise md5 {md5} is not the device's {NOISE_MD5}")
    return noise


def clip_frames(video: torch.Tensor, processor) -> np.ndarray:
    """The decoded clip ([1, 3, F, H, W] in [-1, 1]) as the float [0, 1] frames (F, H, W, 3) that
    the device's VAE stage passes to export_to_video: its processor's ``np`` post-processing."""
    return processor.postprocess_video(video, output_type="np")[0]


def require_fp32(pipe) -> None:
    """Refuse a component that loaded in another dtype: the reference is fp32 end to end (the hub
    stores the Qwen2.5-VL text encoder in bf16; ``torch_dtype`` upcasts it)."""
    for name, module in pipe.components.items():
        if isinstance(module, torch.nn.Module):
            dtypes = {t.dtype for t in (*module.parameters(), *module.buffers())
                      if t.is_floating_point()}
            if dtypes - {torch.float32}:
                raise RuntimeError(f"{name} holds {sorted(map(str, dtypes))}, not fp32 only")


@torch.no_grad()
def run(out_dir: Path, steps: int) -> None:
    from diffusers import HunyuanVideo15Pipeline
    from huggingface_hub import snapshot_download

    start = time.perf_counter()
    pipe = HunyuanVideo15Pipeline.from_pretrained(
        snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True),
        torch_dtype=torch.float32)
    require_fp32(pipe)
    if not pipe.guider._enabled or pipe.guider.guidance_scale != GUIDANCE_SCALE:
        raise RuntimeError(f"the snapshot's guider is not CFG at {GUIDANCE_SCALE}: {pipe.guider}")
    pipe.set_progress_bar_config(disable=True)
    loaded = time.perf_counter()
    cpu = torch.device("cpu")
    names = ("prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2")
    pos = pipe.encode_prompt(prompt=PROMPT, device=cpu, dtype=torch.float32)
    neg = pipe.encode_prompt(prompt=NEGATIVE_PROMPT, device=cpu, dtype=torch.float32)
    embeds = dict(zip(names, pos)) | {f"negative_{n}": t for n, t in zip(names, neg)}
    noise = initial_noise()
    marks = [time.perf_counter()]
    scheduler_step = pipe.scheduler.step

    def timed_step(*args, **kwargs):  # the pipeline has no step callback; one step() per step
        result = scheduler_step(*args, **kwargs)
        marks.append(time.perf_counter())
        print(f"[fp32-ref hunyuan_video_15] step {len(marks) - 1}/{steps} "
              f"{marks[-1] - marks[-2]:.1f} s", flush=True)
        return result

    pipe.scheduler.step = timed_step
    out = pipe(**embeds, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
               num_inference_steps=steps, latents=noise, output_type="latent")
    latents = out.frames.cpu().contiguous()
    pipe.vae.enable_tiling(tile_sample_min_height=VAE_TILE, tile_sample_min_width=VAE_TILE,
                           tile_latent_min_height=VAE_TILE // VAE_SPATIAL,
                           tile_latent_min_width=VAE_TILE // VAE_SPATIAL)
    video = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    decoded = time.perf_counter()
    settings = {"model_id": MODEL_ID, "revision": REVISION, "height": HEIGHT, "width": WIDTH,
                "num_frames": NUM_FRAMES, "steps": steps, "seed": SEED, "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT, "guidance_scale": GUIDANCE_SCALE,
                "vae_tile": VAE_TILE, "cpu_capability": torch.backends.cpu.get_cpu_capability()}
    timings = {"load_s": loaded - start, "encode_s": marks[0] - loaded,
               "steps_s": [b - a for a, b in zip(marks, marks[1:])],
               "denoise_s": marks[-1] - marks[0], "decode_s": decoded - marks[-1],
               "total_s": decoded - start}
    save_run(out_dir, latents=latents, output=clip_frames(video, pipe.video_processor),
             settings=settings, noise_md5=NOISE_MD5, timings=timings, fps=FPS)
    print(f"[fp32-ref hunyuan_video_15] {out_dir}: denoise {timings['denoise_s']:.1f} s, "
          f"total {timings['total_s']:.1f} s", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=96,
                   help="torch threads: the vCPUs of the NUMA node taskset pins the run to")
    p.add_argument("--steps", type=int, default=STEPS, help="50 as on the device; 1 for a smoke")
    args = p.parse_args(argv)
    require_fp32_math()
    set_threads(args.threads)
    run(args.out_dir, args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
