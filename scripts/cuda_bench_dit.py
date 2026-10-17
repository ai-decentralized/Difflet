"""Qwen-Image DiT forward, Difflet vs diffusers, on identical inputs, one GPU.

Three runs per case: warm-up untimed, a clean timed run (median of N forwards
with CUDA events, profiler off) and one profiled forward (``torch.profiler``, kernels bucketed
by ``difflet.backends.cuda.bench``). Cases:

* ``difflet_masked`` — Difflet's real input: ``text_seq_len`` padded text tokens, the first
  ``--valid-text-tokens`` valid (the module may trim the padding itself; the JSON records
  ``trim_text_padding``).
* ``difflet_allones`` — the same tokens with an all-ones mask (Release 1's input).
* ``diffusers_masked`` — upstream ``QwenImageTransformer2DModel`` on the padded tokens + mask.
* ``diffusers_trimmed`` — upstream on the valid tokens only, no mask (what its pipeline feeds).

    DIFFLET_BACKEND=cuda python scripts/cuda_bench_dit.py --model-path <snapshot> [--torch-compile]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

ALL_CASES = ("difflet_masked", "difflet_allones", "diffusers_masked", "diffusers_trimmed")


def _events(prof) -> list[tuple[str, float]]:
    out = []
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", None)
        if us is None:
            us = getattr(e, "self_cuda_time_total", 0.0)
        if us and us > 0:
            out.append((e.key, float(us)))
    return out


def _measure(fn, *, warmup: int, iters: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    from difflet.backends.cuda.bench import summarize_profile, time_forward

    timed = time_forward(fn, warmup=warmup, iters=iters)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    summary = summarize_profile(_events(prof))
    attention = [t["name"] for t in summary["top"] if "attention" == _cat(t["name"])]
    return {"timed": timed, "profile": summary, "attention_kernels": attention[:3]}


def _cat(name: str) -> str:
    from difflet.backends.cuda.bench import categorize_kernel

    return categorize_kernel(name)


def _free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, help="local HF snapshot with transformer/")
    ap.add_argument("--out", default="/tmp/logs/cuda_bench_dit.json")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--text-seq-len", type=int, default=1024)
    ap.add_argument("--valid-text-tokens", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cases", default=",".join(ALL_CASES))
    ap.add_argument("--torch-compile", action="store_true", help="F4 task 3: compiled DiT")
    ap.add_argument("--compile-cache-dir", default="/tmp/logs/cuda_bench_cache")
    args = ap.parse_args()
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tdir = Path(args.model_path) / "transformer"

    from difflet.models.qwen_image.dit_config import QwenImageDiTConfig

    cfg = QwenImageDiTConfig.from_pretrained(
        tdir, height=args.height, width=args.width, text_seq_len=args.text_seq_len
    )
    torch.manual_seed(0)
    hidden = torch.randn(1, cfg.image_seq_len, cfg.in_channels).to(device, dtype)
    timestep = torch.tensor([0.5]).to(device, dtype)
    text = torch.randn(1, cfg.text_seq_len, cfg.joint_attention_dim).to(device, dtype)
    mask_ones = torch.ones(1, cfg.text_seq_len, dtype=torch.bool, device=device)
    mask_part = torch.zeros(1, cfg.text_seq_len, dtype=torch.bool, device=device)
    mask_part[:, : args.valid_text_tokens] = True
    img_shapes = [(1, cfg.packed_height, cfg.packed_width)]

    payload: dict = {
        "model": "qwen_image",
        "model_path": str(args.model_path),
        "shape": [args.height, args.width, args.text_seq_len],
        "valid_text_tokens": args.valid_text_tokens,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_compile": bool(args.torch_compile),
        "warmup": args.warmup,
        "iters": args.iters,
        "cases": {},
    }

    difflet_cases = [c for c in cases if c.startswith("difflet")]
    if difflet_cases:
        from difflet.models.qwen_image.contract import QwenImageDiTInputBundle
        from difflet.models.qwen_image.cuda_application import CudaQwenImageApplication
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        kwargs = {}
        if args.torch_compile:
            kwargs["torch_compile"] = True
        mark = time.monotonic()
        app = CudaQwenImageApplication(
            model_path=str(args.model_path),
            parallel=DiffletParallelConfig(tp_degree=1),
            dtype="bf16",
            shape={"height": args.height, "width": args.width},
            text_seq_len=args.text_seq_len,
            load_text_encoder=False,
            load_vae=False,
            **kwargs,
        )
        app.compile(args.compile_cache_dir)
        app.load(args.compile_cache_dir, skip_warmup=True)
        print(f"difflet loaded in {time.monotonic() - mark:.1f}s", flush=True)
        module = app.transformer.module if hasattr(app, "transformer") else None
        payload["trim_text_padding"] = bool(getattr(module, "trim_text_padding", False))

        def difflet_fn(m):
            bundle = QwenImageDiTInputBundle(
                hidden_states=hidden,
                timestep=timestep,
                encoder_hidden_states=text,
                encoder_hidden_states_mask=m,
                guidance=torch.zeros(1, device=device, dtype=dtype),
            )

            def fn():
                with torch.no_grad():
                    return app(bundle)

            return fn

        for name, m in (("difflet_masked", mask_part), ("difflet_allones", mask_ones)):
            if name not in cases:
                continue
            if args.torch_compile:
                mark = time.monotonic()
                difflet_fn(m)()
                torch.cuda.synchronize()
                first = time.monotonic() - mark
            res = _measure(difflet_fn(m), warmup=args.warmup, iters=args.iters)
            if args.torch_compile:
                res["first_forward_s"] = first
            payload["cases"][name] = res
            print(f"[{name}] {json.dumps(res['timed'])}", flush=True)
        _free(app)

    diffusers_cases = [c for c in cases if c.startswith("diffusers")]
    if diffusers_cases:
        from diffusers import QwenImageTransformer2DModel

        mark = time.monotonic()
        model = QwenImageTransformer2DModel.from_pretrained(str(tdir), torch_dtype=dtype)
        model = model.to(device).eval()
        print(f"diffusers loaded in {time.monotonic() - mark:.1f}s", flush=True)

        def diffusers_fn(t, m):
            def fn():
                with torch.no_grad():
                    return model(
                        hidden_states=hidden,
                        timestep=timestep,
                        encoder_hidden_states=t,
                        encoder_hidden_states_mask=m,
                        img_shapes=img_shapes,
                        return_dict=False,
                    )[0]

            return fn

        for name, t, m in (
            ("diffusers_masked", text, mask_part),
            ("diffusers_trimmed", text[:, : args.valid_text_tokens].contiguous(), None),
        ):
            if name not in cases:
                continue
            res = _measure(diffusers_fn(t, m), warmup=args.warmup, iters=args.iters)
            payload["cases"][name] = res
            print(f"[{name}] {json.dumps(res['timed'])}", flush=True)
        _free(model)

    print(f"\n{'case':<20} {'median ms':>10} {'gemm%':>6} {'attn%':>6} {'elem%':>6}  top kernel")
    for name, res in payload["cases"].items():
        p = res["profile"]
        tot = p["total_ms"] or 1.0
        cat = p["by_category_ms"]
        top = p["top"][0]["name"][:60] if p["top"] else "-"
        print(
            f"{name:<20} {res['timed']['median_ms']:>10.1f} "
            f"{100 * cat.get('gemm', 0) / tot:>6.1f} {100 * cat.get('attention', 0) / tot:>6.1f} "
            f"{100 * cat.get('elementwise', 0) / tot:>6.1f}  {top}"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
