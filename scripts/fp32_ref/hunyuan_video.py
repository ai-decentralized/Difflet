#!/usr/bin/env python3
"""HunyuanVideo's CPU fp32 reference (phase G): diffusers' HunyuanVideoPipeline with every
component in fp32 on the host CPU, at the settings and from the initial noise of the device runs
``score.py`` scores against it (hv1_tp4's bf16 arm and V15, the shipped MXFP8 default).

Every constant is copied from the phase G research object for HunyuanVideo
(``/home/ubuntu/fp8-bench/phase_g_research.json``). What makes this the device's run rather than
diffusers' defaults:

- guidance-distilled: guidance 6.0 enters the DiT's guidance embedder, one forward per step; the
  device has no negative prompt and no true CFG, so ``true_cfg_scale`` stays 1.0;
- the noise is the device's fp32 seed-42 draw (it draws exactly as diffusers does); its md5 is
  checked before it is passed as ``latents``, so a changed law fails instead of being scored;
- the text: diffusers' own ``encode_prompt`` (Llama at 256 tokens, CLIP pooled), which the
  device's rank 0 ran on its host in bf16;
- the decode tiles as the device decoder does (256 pixels, 16 frames, 3/4 strides); untiled,
  diffusers would tile in time only and the pixel metrics would score the tile seams;
- ``latents.pt`` is the last step's latents before ``/ scaling_factor``, [1, 16, 16, 40, 64] fp32,
  the device's layout; the clip is VideoProcessor's frames encoded by ``save_run`` at 24 fps, the
  MP4 the device's ``_write_video`` writes, byte for byte.

The device's own deviations (the DiT timestep and guidance rounded to bf16, the bf16 text encode,
DiT and VAE) are shared by both device arms: they are what the reference measures.

    PYTHONPATH=. python scripts/fp32_ref/hunyuan_video.py --out-dir DIR [--threads 96] [--steps 50]
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

# research device_settings, from the arms' launch scripts (hv1_tp4/run_bf16_arms.sh, run_loo.sh).
MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
REVISION = "e8c2aaa66fe3742a32c11a6766aecbf07c56e773"
HEIGHT, WIDTH, NUM_FRAMES = 320, 512, 61
STEPS, SEED = 50, 42
#: ``--guidance-scale 6.0``: the distilled DiT's embedded guidance (x 1000 in the pipeline).
GUIDANCE_SCALE = 6.0
PROMPT = "a cinematic shot of a red fox running through a snowy forest"
#: research device_settings.negative_prompt: no negative prompt, so no true CFG.
TRUE_CFG_SCALE = 1.0
#: The Llama prompt length the device encodes at (the orchestrator's ``_TEXT_SEQ_LEN``).
TEXT_SEQ_LEN = 256
#: research latents_layout: (B, C, (F - 1) / 4 + 1, H / 8, W / 8), the device's latents.pt.
LATENT_SHAPE = (1, 16, 16, 40, 64)
#: research noise.evidence: difflet's ``_initial_latents`` and diffusers' ``prepare_latents``.
NOISE_MD5 = "fa10d6052fb81005792bbdab0b95d5d3"
#: research reference_recipe (6): NeuronHunyuanVideoVaeDecoderApplication's default windows.
VAE_TILE, VAE_TILE_FRAMES = 256, 16
FPS = DEVICE_FPS["hunyuan_video"]


def initial_noise() -> torch.Tensor:
    """The device's initial latents: the fp32 seed-42 draw (this model draws in fp32, unrounded),
    refused if its md5 is not the one the research recorded for the device."""
    noise = device_noise(LATENT_SHAPE, SEED, bf16_draw=False)
    if tensor_md5(noise) != NOISE_MD5:
        raise RuntimeError(f"initial noise md5 {tensor_md5(noise)} is not the device's {NOISE_MD5}")
    return noise


def clip_frames(video: torch.Tensor, processor) -> np.ndarray:
    """The decoded clip ([1, 3, F, H, W] in [-1, 1]) as the float [0, 1] frames (F, H, W, 3) that
    ``save_run`` encodes: VideoProcessor's ``np`` post-processing, which export_to_video turns
    into the device's ``_video_frames_uint8``."""
    return processor.postprocess_video(video, output_type="np")[0]


def require_fp32(pipe) -> None:
    """Refuse a component that loaded in another dtype: the reference is fp32 end to end (the hub
    stores the DiT in bf16 and the text encoders in fp16; ``torch_dtype`` upcasts them)."""
    for name, module in pipe.components.items():
        if isinstance(module, torch.nn.Module):
            dtypes = {t.dtype for t in (*module.parameters(), *module.buffers())
                      if t.is_floating_point()}
            if dtypes - {torch.float32}:
                raise RuntimeError(f"{name} holds {sorted(map(str, dtypes))}, not fp32 only")


@torch.no_grad()
def run(out_dir: Path, steps: int) -> None:
    from diffusers import HunyuanVideoPipeline
    from huggingface_hub import snapshot_download

    start = time.perf_counter()
    pipe = HunyuanVideoPipeline.from_pretrained(
        snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True),
        torch_dtype=torch.float32)
    require_fp32(pipe)
    pipe.set_progress_bar_config(disable=True)
    loaded = time.perf_counter()
    embeds, pooled, mask = pipe.encode_prompt(prompt=PROMPT, prompt_2=None,
                                              device=torch.device("cpu"),
                                              max_sequence_length=TEXT_SEQ_LEN)
    noise = initial_noise()
    marks = [time.perf_counter()]

    def step_end(_pipe, i, t, kwargs):
        marks.append(time.perf_counter())
        print(f"[fp32-ref hunyuan_video] step {i + 1}/{steps} {marks[-1] - marks[-2]:.1f} s",
              flush=True)
        return kwargs

    out = pipe(prompt_embeds=embeds, pooled_prompt_embeds=pooled, prompt_attention_mask=mask,
               height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES, num_inference_steps=steps,
               guidance_scale=GUIDANCE_SCALE, true_cfg_scale=TRUE_CFG_SCALE, latents=noise,
               output_type="latent", callback_on_step_end=step_end)
    latents = out.frames.cpu().contiguous()
    stride, stride_frames = VAE_TILE * 3 // 4, VAE_TILE_FRAMES * 3 // 4
    pipe.vae.enable_tiling(tile_sample_min_height=VAE_TILE, tile_sample_min_width=VAE_TILE,
                           tile_sample_min_num_frames=VAE_TILE_FRAMES,
                           tile_sample_stride_height=stride, tile_sample_stride_width=stride,
                           tile_sample_stride_num_frames=stride_frames)
    video = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    decoded = time.perf_counter()
    settings = {"model_id": MODEL_ID, "revision": REVISION, "height": HEIGHT, "width": WIDTH,
                "num_frames": NUM_FRAMES, "steps": steps, "seed": SEED, "prompt": PROMPT,
                "guidance_scale": GUIDANCE_SCALE, "true_cfg_scale": TRUE_CFG_SCALE,
                "max_sequence_length": TEXT_SEQ_LEN, "vae_tile": [VAE_TILE, VAE_TILE_FRAMES],
                "cpu_capability": torch.backends.cpu.get_cpu_capability()}
    timings = {"load_s": loaded - start, "encode_s": marks[0] - loaded,
               "steps_s": [b - a for a, b in zip(marks, marks[1:])],
               "denoise_s": marks[-1] - marks[0], "decode_s": decoded - marks[-1],
               "total_s": decoded - start}
    save_run(out_dir, latents=latents, output=clip_frames(video, pipe.video_processor),
             settings=settings, noise_md5=NOISE_MD5, timings=timings, fps=FPS)
    print(f"[fp32-ref hunyuan_video] {out_dir}: denoise {timings['denoise_s']:.1f} s, "
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
