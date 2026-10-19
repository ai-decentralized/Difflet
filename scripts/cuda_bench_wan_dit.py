"""F4b: one Wan 2.2 DiT expert forward, Difflet vs diffusers, on identical inputs, one GPU.

The F4 recipe (``scripts/cuda_bench_dit.py``) for Wan: warm-up untimed, a clean timed run
(median of N forwards with CUDA events, profiler off) and one profiled forward
(``torch.profiler``, kernels bucketed by ``difflet.backends.cuda.bench``). Cases:

* ``difflet`` — ``CudaWanTransformerApplication`` on one expert directory (``--expert
  transformer`` is the high-noise expert, ``transformer_2`` the low-noise one), the module the
  CUDA CLI runs; the inputs are what ``_CudaDiT`` hands it: latents and text in the model
  dtype, the timestep in fp32.
* ``diffusers`` — upstream ``WanTransformer3DModel`` from the same directory on the same
  tensors.

``--cfg-batch 2`` benches the [uncond, cond] stack as one batch-2 call (what ``--cfg-parallel``
scatters); the serial loop makes two batch-1 calls per step instead.

``--loop-probe`` times the orchestrator's host-side work per step without the model: the real
``WanOrchestrator`` denoise loop (UniPC scheduler from the snapshot, expert switch at the
boundary, true CFG) runs with each expert replaced by a probe that moves the inputs to the
device in the model dtype and returns zeros on the host — the copies ``_CudaDiT`` makes, no
compute. ``host_per_step_ms`` is the loop's own cost; next to the forward medians it says
whether the Wan generate gap vs diffusers (F1b: 50.9 s vs 42.9 s for 40 steps) is the DiT or
the loop. The probe runs on a CPU host too (the unit test).

    DIFFLET_BACKEND=cuda python scripts/cuda_bench_wan_dit.py --model-path <snapshot> \\
        [--expert transformer_2] [--cfg-batch 2] [--loop-probe] [--out bench_wan_dit.json]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

ALL_CASES = ("difflet", "diffusers")
EXPERTS = ("transformer", "transformer_2")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True, help="local HF snapshot of Wan 2.2 T2V-A14B")
    ap.add_argument("--expert", choices=EXPERTS, default="transformer")
    ap.add_argument("--cfg-batch", type=int, choices=(1, 2), default=1)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=9)
    ap.add_argument("--text-seq-len", type=int, default=512)
    ap.add_argument("--timestep", type=float, default=999.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cases", default=",".join(ALL_CASES))
    ap.add_argument("--loop-probe", action="store_true", help="time the host loop per step")
    ap.add_argument("--probe-steps", type=int, default=40)
    ap.add_argument("--compile-cache-dir", default="/tmp/logs/cuda_bench_wan_cache")
    ap.add_argument("--out", default="/tmp/logs/bench_wan_dit.json")
    return ap


def bench_inputs(cfg, *, batch: int, timestep: float, device, dtype, seed: int = 0):
    """The expert's inputs as ``_CudaDiT`` hands them: seeded latents ``(B, C, F', H', W')``
    and text ``(B, S, text_dim)`` in the model dtype, the timestep in fp32, one per sample."""
    generator = torch.Generator().manual_seed(int(seed))
    hidden = torch.randn(
        batch,
        int(cfg.in_channels),
        cfg.latent_frames,
        cfg.latent_height,
        cfg.latent_width,
        generator=generator,
    ).to(device, dtype)
    text = torch.randn(batch, int(cfg.text_seq_len), int(cfg.text_dim), generator=generator).to(
        device, dtype
    )
    ts = torch.full((batch,), float(timestep), dtype=torch.float32, device=device)
    return hidden, ts, text


class _ProbeDiT:
    """``_CudaDiT`` without the model: the same copies in and out, zeros back, time recorded."""

    def __init__(self, name: str, cfg, device, dtype, counts: dict, clock: dict) -> None:
        self.name = name
        self.config = cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.timestep_dtype = torch.float32
        self._counts = counts
        self._clock = clock

    def __call__(self, hidden_states, timestep, encoder_hidden_states):
        start = time.perf_counter()
        h = hidden_states.to(self.device, self.dtype)
        timestep.to(self.device, self.timestep_dtype)
        encoder_hidden_states.to(self.device, self.dtype)
        out = torch.zeros_like(h).cpu()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._clock["dit_s"] += time.perf_counter() - start
        self._counts[self.name] = self._counts.get(self.name, 0) + 1
        return out


def loop_probe(
    model_path: str,
    *,
    steps: int,
    device,
    dtype,
    height: int,
    width: int,
    num_frames: int,
    text_seq_len: int,
    guidance_scale: float = 4.0,
    guidance_scale_2: float = 3.0,
) -> dict:
    """The orchestrator's denoise loop with probe experts: per-step host cost, DiT copies
    separated (``dit_ms``), the scheduler and expert split recorded."""
    from difflet.models.wan.dit_config import WanDiTConfig
    from difflet.models.wan.pipeline import WanOrchestrator

    cfg = WanDiTConfig.from_pretrained(
        os.path.join(model_path, "transformer"),
        height=height,
        width=width,
        num_frames=num_frames,
        text_seq_len=text_seq_len,
        torch_dtype=dtype,
    )
    counts: dict[str, int] = {}
    clock = {"dit_s": 0.0}
    experts = {name: _ProbeDiT(name, cfg, device, dtype, counts, clock) for name in EXPERTS}
    orchestrator = WanOrchestrator(
        model_path=model_path,
        transformer=experts["transformer"],
        transformer_2=experts["transformer_2"],
        dtype=dtype,
        height=height,
        width=width,
        num_frames=num_frames,
    )
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(
        1,
        int(cfg.in_channels),
        cfg.latent_frames,
        cfg.latent_height,
        cfg.latent_width,
        generator=generator,
        dtype=torch.float32,
    )
    embeds = torch.zeros(1, int(text_seq_len), int(cfg.text_dim), dtype=dtype)
    start = time.perf_counter()
    orchestrator._denoise(
        latents=latents,
        prompt_embeds=embeds,
        negative_prompt_embeds=torch.zeros_like(embeds),
        num_inference_steps=int(steps),
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
    )
    total_ms = (time.perf_counter() - start) * 1000.0
    dit_ms = clock["dit_s"] * 1000.0
    scheduler = orchestrator.scheduler
    return {
        "steps": int(steps),
        "dit_calls": sum(counts.values()),
        "experts": dict(counts),
        "scheduler": None if scheduler is None else type(scheduler).__name__,
        "latent_shape": list(latents.shape),
        "device": str(device),
        "total_ms": total_ms,
        "dit_ms": dit_ms,
        "host_ms": total_ms - dit_ms,
        "per_step_ms": total_ms / int(steps),
        "host_per_step_ms": (total_ms - dit_ms) / int(steps),
    }


def _cat(name: str) -> str:
    from difflet.backends.cuda.bench import categorize_kernel

    return categorize_kernel(name)


def _measure(fn, *, warmup: int, iters: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    from difflet.backends.cuda.bench import kernel_events, summarize_profile, time_forward

    timed = time_forward(fn, warmup=warmup, iters=iters)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    summary = summarize_profile(kernel_events(prof.key_averages()))
    attention = [t["name"] for t in summary["top"] if _cat(t["name"]) == "attention"]
    return {"timed": timed, "profile": summary, "attention_kernels": attention[:3]}


def _free() -> None:
    """After the caller dropped its references (the model and any closure over it)."""
    gc.collect()
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    expert_dir = Path(args.model_path) / args.expert

    from difflet.models.wan.dit_config import WanDiTConfig

    cfg = WanDiTConfig.from_pretrained(
        expert_dir,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        text_seq_len=args.text_seq_len,
        batch_size=args.cfg_batch,
        torch_dtype=dtype,
    )
    hidden, timestep, text = bench_inputs(
        cfg, batch=args.cfg_batch, timestep=args.timestep, device=device, dtype=dtype
    )
    payload: dict = {
        "model": "wan",
        "model_path": str(args.model_path),
        "expert": args.expert,
        "cfg_batch": args.cfg_batch,
        "shape": [args.height, args.width, args.num_frames, args.text_seq_len],
        "latent_shape": list(hidden.shape),
        "timestep": args.timestep,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "cases": {},
    }

    if "difflet" in cases:
        from difflet.backends.cuda.wan.transformer import CudaWanTransformerApplication

        mark = time.monotonic()
        torch_compile = os.environ.get("DIFFLET_CUDA_TORCH_COMPILE", "") == "1"
        payload["torch_compile"] = torch_compile
        app = CudaWanTransformerApplication(
            model_path=str(expert_dir), config=cfg, device=device, torch_compile=torch_compile
        )
        cache = str(Path(args.compile_cache_dir) / args.expert)
        app.compile(cache)
        app.load(cache, skip_warmup=True)
        print(f"difflet {args.expert} loaded in {time.monotonic() - mark:.1f}s", flush=True)

        def difflet_fn():
            with torch.no_grad():
                return app(hidden, timestep, text)

        res = _measure(difflet_fn, warmup=args.warmup, iters=args.iters)
        payload["cases"]["difflet"] = res
        print(f"[difflet] {json.dumps(res['timed'])} attention={res['attention_kernels']}")
        # Drop every reference (the closure too) so the expert leaves the card before the
        # diffusers side loads; two experts fit a 96 GB card, which hid a no-op here (review).
        del difflet_fn
        app = None
        _free()

    if "diffusers" in cases:
        from diffusers import WanTransformer3DModel

        mark = time.monotonic()
        model = WanTransformer3DModel.from_pretrained(str(expert_dir), torch_dtype=dtype)
        model = model.to(device).eval()
        print(f"diffusers {args.expert} loaded in {time.monotonic() - mark:.1f}s", flush=True)

        def diffusers_fn():
            with torch.no_grad():
                return model(
                    hidden_states=hidden,
                    timestep=timestep,
                    encoder_hidden_states=text,
                    return_dict=False,
                )[0]

        res = _measure(diffusers_fn, warmup=args.warmup, iters=args.iters)
        payload["cases"]["diffusers"] = res
        print(f"[diffusers] {json.dumps(res['timed'])} attention={res['attention_kernels']}")
        del diffusers_fn
        model = None
        _free()

    if args.loop_probe:
        payload["loop_probe"] = loop_probe(
            str(args.model_path),
            steps=args.probe_steps,
            device=device,
            dtype=dtype,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            text_seq_len=args.text_seq_len,
        )
        print(f"[loop-probe] {json.dumps(payload['loop_probe'])}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
