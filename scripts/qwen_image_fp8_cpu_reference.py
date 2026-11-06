#!/usr/bin/env python3
"""CPU arbiter for Qwen-Image MXFP8: the bf16 model against its fake-quantized twin, no device.

Runs diffusers' ``QwenImagePipeline`` on the host twice per prompt from the same seed: once in
bf16, once with the target linears replaced by ``FakeQuantMXLinear``
(``difflet/quant/fake_linear.py``: fp8 weights with block scales, the activation block-quantized
per call, fp32 matmuls, exactly the device kernel's arithmetic). The difference between the two
is the quantization error alone, with no device in the loop, so it decides whether the recipe is
worth putting on a chip (gate: PSNR >= 30 dB on every prompt).

Writes under ``--out``: ``spec.json``, per prompt ``<i>_bf16.png`` / ``<i>_fp8.png`` and their
packed latents, ``metrics.json`` and ``summary.md``.

    PYTHONPATH=. OMP_NUM_THREADS=96 python scripts/qwen_image_fp8_cpu_reference.py \\
        --out /home/ubuntu/fp8-bench/cpu_arbiter --steps 20 --tp 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from difflet.quant.fake_linear import fake_quantize_model_  # noqa: E402
from difflet.quant.targets import default_spec  # noqa: E402

MODEL_ID = "Qwen/Qwen-Image"
PROMPTS = [
    "a cat sitting on a bench",
    "a watercolor painting of a lighthouse on a rocky coast at dawn, soft light, detailed",
    "a close-up studio photograph of a vintage brass typewriter on a wooden desk",
    "an astronaut riding a bicycle through a neon-lit city street at night, cinematic",
]
GATE_PSNR_DB = 30.0


def _metrics_module():
    spec = importlib.util.spec_from_file_location("qvr", REPO / "scripts" / "quality_vs_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(model_dir: Path) -> dict:
    parts = model_dir.resolve().parts
    for i, part in enumerate(parts):
        if part.startswith("models--") and i + 2 < len(parts) and parts[i + 1] == "snapshots":
            return {"model_id": MODEL_ID, "snapshot": parts[i + 2], "path": str(model_dir)}
    return {"model_id": MODEL_ID, "path": str(model_dir)}


def _generate(pipe, prompt: str, *, seed: int, steps: int, height: int, width: int):
    """Packed latents and the decoded image, from the seeded noise."""
    out = pipe(prompt=prompt, negative_prompt=" ", true_cfg_scale=1.0,
               num_inference_steps=steps, height=height, width=width,
               generator=torch.Generator().manual_seed(seed), output_type="latent")
    latents = out.images  # packed [B, S, 64]
    latents_unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents_unpacked = latents_unpacked.to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    latents_unpacked = latents_unpacked / std.to(latents_unpacked.dtype) + mean.to(latents_unpacked.dtype)
    with torch.no_grad():
        image = pipe.vae.decode(latents_unpacked, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return latents.detach().float().cpu(), image


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir", default=None, help="local snapshot root (default: HF cache)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompts", nargs="*", default=PROMPTS)
    p.add_argument("--tp", type=int, default=4,
                   help="the TP degree whose row-parallel K shards the blocks follow")
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qvr = _metrics_module()

    from diffusers import QwenImagePipeline
    from huggingface_hub import snapshot_download

    model_dir = Path(args.model_dir or snapshot_download(MODEL_ID))
    print(f"[arbiter] model {model_dir}", flush=True)
    t0 = time.perf_counter()
    pipe = QwenImagePipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False)
    print(f"[arbiter] loaded in {time.perf_counter() - t0:.0f}s; threads={torch.get_num_threads()}",
          flush=True)

    spec = default_spec("qwen_image", tp_degree=args.tp)
    (out / "spec.json").write_text(json.dumps(spec.to_dict(), indent=1) + "\n")
    results: list[dict] = []

    # ---- arm A: bf16
    bf16 = {}
    for i, prompt in enumerate(args.prompts):
        t = time.perf_counter()
        latents, image = _generate(pipe, prompt, seed=args.seed, steps=args.steps,
                                   height=args.height, width=args.width)
        bf16[i] = (latents, image)
        torch.save(latents, out / f"{i}_bf16_latents.pt")
        image.save(out / f"{i}_bf16.png")
        print(f"[arbiter] bf16 prompt {i}: {time.perf_counter() - t:.0f}s, "
              f"{(time.perf_counter() - t) / args.steps:.1f}s/step", flush=True)

    # ---- arm B: fake-quant mxfp8 on the same seeds
    replaced = fake_quantize_model_(pipe.transformer, spec)
    print(f"[arbiter] fake-quantized {len(replaced)} linears", flush=True)
    for i, prompt in enumerate(args.prompts):
        t = time.perf_counter()
        latents, image = _generate(pipe, prompt, seed=args.seed, steps=args.steps,
                                   height=args.height, width=args.width)
        torch.save(latents, out / f"{i}_fp8_latents.pt")
        image.save(out / f"{i}_fp8.png")
        ref_latents, ref_image = bf16[i]
        lat = qvr.latent_metrics(ref_latents, latents)
        frames = qvr.frame_metrics(np.asarray(ref_image)[None, ..., :3], np.asarray(image)[None, ..., :3])
        row = {"prompt": prompt,
               "latent_cosine": lat["cosine"], "latent_rel_l2": lat["rel_l2"],
               "psnr": frames["psnr_mean"], "ssim": frames["ssim_mean"],
               "fp8_seconds": time.perf_counter() - t}
        results.append(row)
        print(f"[arbiter] mxfp8 prompt {i}: PSNR {row['psnr']:.2f} dB  SSIM {row['ssim']:.4f}  "
              f"latent cos {row['latent_cosine']:.4f} rel_l2 {row['latent_rel_l2']:.4f}  "
              f"({row['fp8_seconds']:.0f}s)", flush=True)

    passed = all(r["psnr"] >= GATE_PSNR_DB for r in results)
    summary = {"model": _snapshot(model_dir), "steps": args.steps, "seed": args.seed,
               "shape": [args.height, args.width], "spec": spec.to_dict(),
               "gate_psnr_db": GATE_PSNR_DB, "passed": passed,
               "results": results}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = ["# Qwen-Image MXFP8 CPU arbiter", "",
             f"bf16 vs fake-quant mxfp8 (block-32 MX scales, tp {spec.tp_degree}), "
             f"{args.steps} steps, {args.height}x{args.width}, seed "
             f"{args.seed}, true_cfg 1.0, CPU bf16. Gate: PSNR >= {GATE_PSNR_DB:g} dB.", "",
             "| # | prompt | PSNR dB | SSIM | latent cos | latent rel L2 |",
             "|---|---|---:|---:|---:|---:|"]
    for i, r in enumerate(results):
        lines.append(f"| {i} | {r['prompt'][:48]} | "
                     f"{r['psnr']:.2f} | {r['ssim']:.4f} | {r['latent_cosine']:.4f} | "
                     f"{r['latent_rel_l2']:.4f} |")
    lines += ["", f"**{'PASS' if passed else 'FAIL'}**: "
              f"min PSNR {min(r['psnr'] for r in results):.2f} dB"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
