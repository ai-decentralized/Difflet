"""HunyuanVideo 1.5's DiT application on the device: parity against an fp32 diffusers oracle, and
time per forward, in compile mode at TP1.

Phase 1, CPU: a 5-frame 480x848 clip (2 x 30 x 53 = 3,180 image tokens) at the pipeline's real text
lengths (Qwen2.5-VL 1,000 tokens with 120 valid, ByT5 256 empty, 729 zero image embeddings);
diffusers in fp32 (reference) and bf16 (control).

    python tests/manual/check_neuron_hv15_dit.py --model-dir $HV15 --make-oracle ~/w22/hv15_oracle.pt

Phase 2, device: the application, compile mode, one forward compared, then --iters timed. One
rank is TP1; N ranks is TP1 x CP=N (Ulysses: the image tokens split, the text replicated).
``--oracle none --frames F``: timing only at another length (finite output, time per forward).

    torchrun --standalone --nproc-per-node 1 tests/manual/check_neuron_hv15_dit.py \
        --model-dir $HV15 --oracle ~/w22/hv15_oracle.pt --iters 5
    torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_hv15_dit.py \
        --model-dir $HV15 --oracle ~/w22/hv15_oracle.pt --iters 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

GATE = 1.15
H, W, FRAMES = 480, 848, 5


def _inputs(frames: int = FRAMES):
    g = torch.Generator().manual_seed(0)
    mask = torch.zeros(1, 1000, dtype=torch.int64)
    mask[:, :120] = 1
    lat = (frames - 1) // 4 + 1, H // 16, W // 16
    return dict(
        hidden_states=torch.cat([torch.randn(1, 32, *lat, generator=g), torch.zeros(1, 33, *lat)], dim=1),
        timestep=torch.tensor([900.0]),
        encoder_hidden_states=torch.randn(1, 1000, 3584, generator=g),
        encoder_attention_mask=mask,
    )


def _t2v_extras(dtype):
    return dict(encoder_hidden_states_2=torch.zeros(1, 256, 1472, dtype=dtype),
                encoder_attention_mask_2=torch.zeros(1, 256, dtype=torch.int64),
                image_embeds=torch.zeros(1, 729, 1152, dtype=dtype))


def make_oracle(args) -> int:
    from diffusers import HunyuanVideo15Transformer3DModel

    path = os.path.join(args.model_dir, "transformer")
    out = {"inputs": _inputs()}
    for name, dtype in (("ref_fp32", torch.float32), ("control_bf16", torch.bfloat16)):
        t = time.perf_counter()
        model = HunyuanVideo15Transformer3DModel.from_pretrained(path, torch_dtype=dtype).eval()
        x = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in out["inputs"].items()}
        with torch.no_grad():
            out[name] = model(**x, **_t2v_extras(dtype), return_dict=False)[0].float()
        del model
        print(f"{name}: {time.perf_counter() - t:.0f} s, finite={bool(torch.isfinite(out[name]).all())}", flush=True)
    torch.save(out, os.path.expanduser(args.make_oracle))
    return 0


def check(args) -> int:
    from difflet.backends.neuron.hunyuan_video15.transformer import NeuronHunyuanVideo15TransformerApplication
    from difflet.backends.registry import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    oracle = None if args.oracle == "none" else torch.load(os.path.expanduser(args.oracle))
    frames = FRAMES if oracle is not None else int(args.frames)
    world, tp = int(os.environ.get("WORLD_SIZE", "1")), int(args.tp_degree)
    cfg = 2 if args.cfg_parallel else 1
    if world % (tp * cfg):
        raise SystemExit(f"world size {world} is not divisible by tp x cfg = {tp * cfg}")
    cp = world // (tp * cfg)
    parallel = DiffletParallelConfig(tp_degree=tp, cp_degree=cp, cfg_parallel_enabled=cfg > 1,
                                     cp_mode=args.cp_mode if cp > 1 else "gather_kv")
    layout = f"TP{tp}" + (f" x CP{cp} ({args.cp_mode})" if cp > 1 else "") + (" x CFG2" if cfg > 1 else "")
    get_backend("neuron").prepare_runtime(parallel)  # core binding + the numerics workarounds
    import torch.distributed as dist

    rank0 = not dist.is_initialized() or dist.get_rank() == 0
    say = (lambda *a, **k: print(*a, **k)) if rank0 else (lambda *a, **k: None)  # noqa: E731
    app = NeuronHunyuanVideo15TransformerApplication(
        model_path=os.path.join(args.model_dir, "transformer"), height=H, width=W, num_frames=frames,
        parallel=parallel, dtype=torch.bfloat16, exec_mode=args.exec_mode)
    t = time.perf_counter()
    app.load()
    say(f"load+warmup {time.perf_counter() - t:.1f} s phases="
          f"{ {k: round(v, 1) for k, v in app.phase_seconds.items()} } report={app.load_report}", flush=True)

    inputs = oracle["inputs"] if oracle is not None else _inputs(frames)
    if cfg > 1:  # both branches get the same inputs: each half must reproduce the reference
        inputs = {k: torch.cat([v, v]) for k, v in inputs.items()}

    def forward():
        out = app(**inputs)[0]
        torch.neuron.synchronize()
        return out

    y = forward().float().cpu()
    halves = [y[i:i + 1] for i in range(y.shape[0])] if cfg > 1 else [y]
    y = halves[0]
    if oracle is None:  # timing only: no reference at this length
        say(f"{frames} frames ({app.num_image_tokens} image tokens), {layout}: "
            f"finite={bool(torch.isfinite(y).all())}, output {tuple(y.shape)}", flush=True)
        times = []
        for _ in range(args.iters):
            t = time.perf_counter()
            forward()
            times.append(time.perf_counter() - t)
        if times:
            say(f"forward median {statistics.median(times) * 1e3:.1f} ms (n={len(times)})", flush=True)
        return 0
    ref, ctl = oracle["ref_fp32"], oracle["control_bf16"]
    rel = lambda a: ((a - ref).abs().mean() / ref.abs().mean()).item()  # noqa: E731
    finite = all(bool(torch.isfinite(h).all()) for h in halves)
    ours = max(rel(h) for h in halves) if finite else float("nan")
    control = rel(ctl)
    ok = finite and ours <= GATE * control
    say(f"{'PASS' if ok else 'FAIL'} {args.exec_mode} {layout}: finite={finite} rel_l1={ours:.3e} "
          f"({ours / control:.3f}x control {control:.3e}, gate {GATE}x)", flush=True)
    times = []
    for _ in range(args.iters):
        t = time.perf_counter()
        forward()
        times.append(time.perf_counter() - t)
    if times:
        say(f"forward median {statistics.median(times) * 1e3:.1f} ms (n={len(times)})", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--make-oracle")
    p.add_argument("--oracle")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--exec-mode", choices=("compile", "eager"), default="compile")
    p.add_argument("--tp-degree", type=int, default=1, help="TP degree; CP takes the rest of the world")
    p.add_argument("--cp-mode", choices=("ulysses", "ring", "gather_kv"), default="ulysses")
    p.add_argument("--cfg-parallel", action="store_true",
                   help="two cfg groups, one guidance branch each (the parity check covers both halves)")
    p.add_argument("--frames", type=int, default=FRAMES,
                   help="with --oracle none: the clip length for a timing-only run")
    args = p.parse_args()
    if args.make_oracle:
        return make_oracle(args)
    if args.oracle:
        return check(args)
    p.error("give --make-oracle or --oracle")


if __name__ == "__main__":
    sys.exit(main())
