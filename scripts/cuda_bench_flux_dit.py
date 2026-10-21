"""One FLUX.1-dev DiT forward, Difflet vs diffusers, on identical inputs, one GPU.

The F4b recipe (``scripts/cuda_bench_wan_dit.py``) for Flux: warm-up untimed, a clean timed run
(median of N forwards with CUDA events, profiler off) and one profiled forward
(``torch.profiler``, kernels bucketed by ``difflet.backends.cuda.bench``). Cases:

* ``difflet`` — ``CudaFluxTransformerApplication`` on the snapshot's ``transformer/`` (the
  module the CUDA CLI runs), fed what ``FluxDiTAdapter`` hands it: packed latents, text,
  pooled projections, ``t / 1000`` and guidance in the model dtype, and the stacked
  ``(S, head_dim, 2)`` rope the adapter builds from the pipeline's ``txt_ids`` / ``img_ids``.
* ``diffusers`` — upstream ``FluxTransformer2DModel`` on the same tensors, with the ids.

    DIFFLET_BACKEND=cuda python scripts/cuda_bench_flux_dit.py --model-path <snapshot> \\
        [--out bench_flux_dit.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuda_bench_wan_dit import _free, _measure  # noqa: E402

ALL_CASES = ("difflet", "diffusers")
TEXT_SEQ_LEN = 512


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True, help="local HF snapshot of FLUX.1-dev")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--timestep", type=float, default=0.5, help="the pipeline's t / 1000")
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cases", default=",".join(ALL_CASES))
    ap.add_argument("--compile-cache-dir", default="/tmp/logs/cuda_bench_flux_cache")
    ap.add_argument("--out", default="/tmp/logs/bench_flux_dit.json")
    ap.add_argument(
        "--torch-compile",
        action="store_true",
        help="compile the DiT blocks, as `difflet run --torch-compile` does (default: eager)",
    )
    return ap


def latent_image_ids(height: int, width: int, *, device, dtype) -> torch.Tensor:
    """``FluxPipeline._prepare_latent_image_ids`` over the ``H/16 x W/16`` patch grid."""
    rows, cols = height // 16, width // 16
    ids = torch.zeros(rows, cols, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(rows)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(cols)[None, :]
    return ids.reshape(rows * cols, 3).to(device=device, dtype=dtype)


def bench_inputs(cfg, *, timestep: float, guidance: float, device, dtype, seed: int = 0) -> dict:
    """The DiT's inputs as the pipeline hands them to both models: seeded packed latents, text
    and pooled projections in the model dtype, ``t / 1000`` and guidance as ``(1,)`` tensors
    (empty guidance for a schnell-shaped config), and the ids the rope is built from."""
    generator = torch.Generator().manual_seed(int(seed))
    hidden = torch.randn(1, int(cfg.image_seq_len), int(cfg.in_channels), generator=generator)
    text = torch.randn(1, TEXT_SEQ_LEN, int(cfg.joint_attention_dim), generator=generator)
    pooled = torch.randn(1, int(cfg.pooled_projection_dim), generator=generator)
    guidance_t = (
        torch.tensor([float(guidance)], dtype=dtype, device=device)
        if cfg.guidance_embeds
        else torch.tensor([], dtype=dtype, device=device)
    )
    return {
        "hidden_states": hidden.to(device, dtype),
        "encoder_hidden_states": text.to(device, dtype),
        "pooled_projections": pooled.to(device, dtype),
        "timestep": torch.tensor([float(timestep)], dtype=dtype, device=device),
        "guidance": guidance_t,
        "txt_ids": torch.zeros(TEXT_SEQ_LEN, 3, device=device, dtype=torch.float32),
        "img_ids": latent_image_ids(cfg.height, cfg.width, device=device, dtype=torch.float32),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tdir = Path(args.model_path) / "transformer"

    from difflet.models.flux.dit_config import FluxDiTConfig

    cfg = FluxDiTConfig.from_pretrained(
        tdir, height=args.height, width=args.width, tp_degree=1, torch_dtype=dtype
    )
    inputs = bench_inputs(
        cfg, timestep=args.timestep, guidance=args.guidance, device=device, dtype=dtype
    )
    payload: dict = {
        "model": "flux",
        "model_path": str(args.model_path),
        "shape": [args.height, args.width],
        "image_seq_len": int(cfg.image_seq_len),
        "timestep": args.timestep,
        "guidance": args.guidance,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "cases": {},
    }

    if "difflet" in cases:
        from difflet.backends.cuda.flux.transformer import (
            CudaFluxTransformerApplication,
            FluxDiTAdapter,
        )

        mark = time.monotonic()
        torch_compile = bool(args.torch_compile)
        payload["torch_compile"] = torch_compile
        app = CudaFluxTransformerApplication(
            model_path=str(tdir), config=cfg, device=device, torch_compile=torch_compile
        )
        app.compile(args.compile_cache_dir)
        app.load(args.compile_cache_dir, skip_warmup=True)
        print(f"difflet loaded in {time.monotonic() - mark:.1f}s", flush=True)
        rope = FluxDiTAdapter(app).rotary_embedding(inputs["txt_ids"], inputs["img_ids"])
        payload["rope_shape"] = list(rope.shape)

        def difflet_fn():
            with torch.no_grad():
                return app(
                    inputs["hidden_states"],
                    inputs["encoder_hidden_states"],
                    inputs["pooled_projections"],
                    inputs["timestep"],
                    inputs["guidance"],
                    rope,
                )

        res = _measure(difflet_fn, warmup=args.warmup, iters=args.iters)
        payload["cases"]["difflet"] = res
        print(f"[difflet] {json.dumps(res['timed'])} attention={res['attention_kernels']}")
        del difflet_fn
        app = None
        rope = None
        _free()

    if "diffusers" in cases:
        from diffusers import FluxTransformer2DModel

        mark = time.monotonic()
        model = FluxTransformer2DModel.from_pretrained(str(tdir), torch_dtype=dtype)
        model = model.to(device).eval()
        print(f"diffusers loaded in {time.monotonic() - mark:.1f}s", flush=True)
        guidance = inputs["guidance"] if inputs["guidance"].numel() else None

        def diffusers_fn():
            with torch.no_grad():
                return model(
                    hidden_states=inputs["hidden_states"],
                    encoder_hidden_states=inputs["encoder_hidden_states"],
                    pooled_projections=inputs["pooled_projections"],
                    timestep=inputs["timestep"],
                    img_ids=inputs["img_ids"],
                    txt_ids=inputs["txt_ids"],
                    guidance=guidance,
                    return_dict=False,
                )[0]

        res = _measure(diffusers_fn, warmup=args.warmup, iters=args.iters)
        payload["cases"]["diffusers"] = res
        print(f"[diffusers] {json.dumps(res['timed'])} attention={res['attention_kernels']}")
        del diffusers_fn
        model = None
        _free()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
