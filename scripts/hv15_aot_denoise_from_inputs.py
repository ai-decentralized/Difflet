#!/usr/bin/env python3
"""Run a compiled HunyuanVideo 1.5 AOT DiT (``difflet compile`` on the trainium backend) on given
inputs: an ``embeds.pt`` from any encode stage and, optionally, a saved initial noise tensor.

This is the AOT transformer stage's generate loop (``_stage_aot_transformer``) with the inputs made
explicit, so the AOT DiT can be compared with the native backend on bit-identical inputs. The two
backends run on different PyTorch stacks, and the same seed draws different bf16 noise there
(torch 2.9 samples bf16 directly; 2.14 samples fp32 and rounds), so ``--noise-seed-fp32-round``
draws the noise the way the native stack does.

Launch with the AOT runtime's environment (one process drives all the layout's cores):
NEURON_RT_VIRTUAL_CORE_SIZE=2, and NEURON_RT_VISIBLE_CORES=0..n-1 (n = tp x cfg x cp) when n is
below the device's core count and not a multiple of 8 (NRT 2.35 rejects NEURON_RT_NUM_CORES there).

    python scripts/hv15_aot_denoise_from_inputs.py --compiled-dir DIR --embeds embeds.pt \
        --tp-degree 2 --height 320 --width 512 --num-frames 61 --steps 3 --noise-seed-fp32-round 42 \
        --out latents.pt
"""
from __future__ import annotations

import argparse
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    p.add_argument("--revision", default=None)
    p.add_argument("--compiled-dir", required=True)
    p.add_argument("--embeds", required=True)
    p.add_argument("--tp-degree", type=int, default=2)
    p.add_argument("--cp-degree", type=int, default=1)
    p.add_argument("--cp-mode", default=None)
    p.add_argument("--cfg-parallel", action="store_true")
    p.add_argument("--sp", action="store_true")
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--num-frames", type=int, required=True)
    p.add_argument("--steps", type=int, default=50)
    noise = p.add_mutually_exclusive_group()
    noise.add_argument("--noise", default=None, help="a saved initial-noise tensor")
    noise.add_argument("--noise-seed-fp32-round", type=int, default=None,
                       help="draw fp32 noise with this seed and round it to bf16 (the native stack's draw)")
    p.add_argument("--seed", type=int, default=42, help="the generator seed when no noise is given")
    p.add_argument("--save-noise", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import torch
    from diffusers.utils.torch_utils import randn_tensor

    from difflet.models.hunyuan_video import aot15
    from difflet.models.hunyuan_video.pipeline15 import load_hv15_guidance
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from difflet.pipeline.path_resolver import resolve_model_path

    root = resolve_model_path(a.model_id, revision=a.revision, local_files_only=True)
    cp_mode = (a.cp_mode or "ulysses") if a.cp_degree > 1 else "gather_kv"
    parallel = DiffletParallelConfig(tp_degree=a.tp_degree, cp_degree=a.cp_degree, cfg_parallel_enabled=a.cfg_parallel,
                                     sp_enabled=a.sp, cp_mode=cp_mode)
    started = time.perf_counter()
    denoiser = aot15.build_aot_denoiser(model_path=root, compiled_model_path=a.compiled_dir, parallel=parallel,
                                        height=a.height, width=a.width, num_frames=a.num_frames)
    denoiser.load(skip_warmup=True)
    print(f"[hv15-inputs] load {time.perf_counter() - started:.1f} s", flush=True)
    denoiser.guidance_scale = load_hv15_guidance(root, None)
    shape = (1, int(denoiser.dit.model_config["out_channels"]), (a.num_frames - 1) // 4 + 1, a.height // 16,
             a.width // 16)
    latents = None
    if a.noise:
        latents = torch.load(a.noise, map_location="cpu", weights_only=True)
    elif a.noise_seed_fp32_round is not None:
        latents = randn_tensor(shape, generator=torch.Generator().manual_seed(a.noise_seed_fp32_round),
                               device=torch.device("cpu"), dtype=torch.float32).to(torch.bfloat16)
    if latents is not None and a.save_noise:
        torch.save(latents, a.save_noise)
    embeds = torch.load(a.embeds, map_location="cpu", weights_only=True)
    marks = [time.perf_counter()]
    out = denoiser(**embeds, num_inference_steps=a.steps, generator=torch.Generator().manual_seed(a.seed),
                   latents=latents, callback=lambda i, t, x: marks.append(time.perf_counter()))
    steps = sorted(b - c for c, b in zip(marks, marks[1:]))
    print(f"[hv15-inputs] denoise {marks[-1] - marks[0]:.1f} s, {len(steps)} steps, "
          f"median {steps[len(steps) // 2] * 1e3:.0f} ms (first {(marks[1] - marks[0]) * 1e3:.0f} ms)", flush=True)
    torch.save(out, a.out)
    print(f"[hv15-inputs] latents saved to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
