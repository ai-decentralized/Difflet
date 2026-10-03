"""Compile and verify the streaming Wan VAE with a pinned real checkpoint.

Run from the repository root with the Neuron venv activated, e.g.:
  python -m scripts.benchmark_wan_vae_chunked --model-path SNAPSHOT \
      --output-dir artifacts/wan-vae-chunked/run-01 --height 480 --width 832

The output directory must be new unless --resume-compile or --load-only is
used. Each attempt writes a new verification receipt. This is a VAE-only synthetic-latent test,
not a full text-to-video quality or end-to-end performance measurement.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--reference-frames", type=int, default=9)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--resume-compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()
    if args.compile_only and args.load_only:
        parser.error("--compile-only and --load-only are mutually exclusive")
    if args.resume_compile and args.load_only:
        parser.error("--resume-compile and --load-only are mutually exclusive")
    for frames in (args.num_frames, args.reference_frames):
        if frames < 1 or frames % 4 != 1:
            parser.error("Frame counts must be positive and of the form 4*k+1")
    if args.reference_frames > args.num_frames or args.iters < 1:
        parser.error("Reference length must not exceed the request; iters must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=args.load_only or args.resume_compile)
    out = args.output_dir.resolve()
    os.environ.setdefault("BASE_COMPILE_WORK_DIR", str(out / "compiler"))
    os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    import torch
    from difflet.models.wan.application import create_wan_vae_decoder_config
    from difflet.backends.trainium.wan.vae import NeuronWanVAEChunkedApplication

    torch.set_num_threads(4)
    config = create_wan_vae_decoder_config(
        model_path=args.model_path, world_size=1, tp_degree=1, dtype=torch.float32,
        height=args.height, width=args.width, num_frames=args.num_frames,
    )
    app = NeuronWanVAEChunkedApplication(model_path=str(Path(args.model_path) / "vae"), config=config)
    source_paths = [
        "difflet/models/wan/vae/chunked.py", "difflet/models/wan/vae/modeling_vae.py",
        "difflet/backends/trainium/wan/vae.py", "scripts/benchmark_wan_vae_chunked.py",
        "difflet/backends/trainium/utils/compile_serial.py",
    ]
    receipt = {
        "kind": "vae_only_synthetic_latents", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "diffusers", "torch-neuronx", "neuronx-cc", "neuronx-distributed")},
        "sources": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths},
        "status": "running",
        "compiler_work_dir": os.environ["BASE_COMPILE_WORK_DIR"],
    }
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt_path = out / f"receipt-{stamp}.json"

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        artifact = out / "compiled"
        if not args.load_only:
            start = time.perf_counter()
            app.compile(str(artifact))
            receipt["compile_seconds"] = time.perf_counter() - start
            save()
            print(f"Compiled in {receipt['compile_seconds']:.1f}s", flush=True)
        if not args.compile_only:
            start = time.perf_counter()
            app.load(str(artifact), start_rank_id=0, local_ranks_size=1, skip_warmup=True)
            receipt["load_seconds"] = time.perf_counter() - start
            z = torch.randn(
                1, config.z_dim, (args.num_frames - 1) // 4 + 1, args.height // 8, args.width // 8,
                generator=torch.Generator().manual_seed(args.seed), dtype=torch.float32,
            )
            # Model consumes denormalized VAE latents, matching WanPipeline._decode_latents.
            mean = torch.tensor(config.latents_mean).view(1, -1, 1, 1, 1)
            std = torch.tensor(config.latents_std).view(1, -1, 1, 1, 1)
            z = z * std + mean
            with torch.no_grad():
                actual = app(z).cpu()  # discard warm-up
                samples = []
                for _ in range(args.iters):
                    start = time.perf_counter()
                    repeated = app(z).cpu()
                    samples.append(time.perf_counter() - start)
                    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
                assert tuple(actual.shape) == (1, 3, args.num_frames, args.height, args.width)
                assert torch.isfinite(actual).all()
                receipt.update(decode_seconds=samples, output_shape=list(actual.shape), finite=True, repeated_bitwise=True)
                save()
                print(f"Decode seconds: {samples}; checking {args.reference_frames} reference frames", flush=True)
                from diffusers import AutoencoderKLWan
                reference = AutoencoderKLWan.from_pretrained(
                    str(Path(args.model_path) / "vae"), torch_dtype=torch.float32,
                ).eval()
                ref_latents = (args.reference_frames - 1) // 4 + 1
                expected = reference.decode(z[:, :, :ref_latents], return_dict=False)[0].clamp(-1, 1)
                measured = actual[:, :, :args.reference_frames]
                error = (measured - expected).float()
                receipt.update(reference_frames=args.reference_frames, max_abs=float(error.abs().max()),
                               rmse=float(error.square().mean().sqrt()),
                               relative_l2=float(error.norm() / expected.norm().clamp_min(1e-12)))
                save()
                # FP32 accelerator arithmetic can differ; retain raw errors and enforce a fixed gate.
                if receipt["relative_l2"] > 0.01 or receipt["max_abs"] > 0.1:
                    raise AssertionError("VAE reference mismatch: relL2 > 0.01 or max_abs > 0.1")
        receipt["status"] = "passed"
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
        print(f"Receipt: {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
