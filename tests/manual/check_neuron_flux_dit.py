"""Flux DiT on the neuron backend: parity against an fp32 diffusers oracle, and time per forward.

Phase 1, CPU only (no torchrun): fixed inputs at the static shape, diffusers'
FluxTransformer2DModel in fp32 (the reference) and in bf16 (the control), saved together.

    python tests/manual/check_neuron_flux_dit.py --model-dir $FLUX --make-oracle ~/w22/flux_oracle.pt

Phase 2, on the device under torchrun: the neuron application at TP=world (compile mode), one
forward on the oracle's inputs compared on rank 0, then --iters timed forwards (FLUX.1-dev is
guidance-distilled: one forward per denoising step).

    torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_flux_dit.py \
        --model-dir $FLUX --oracle ~/w22/flux_oracle.pt --iters 10

Gate (as for Wan): rel_l1(ours, fp32) <= 1.15 x rel_l1(bf16 control, fp32).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

GATE = 1.15


def _inputs(config, seed: int = 0) -> dict:
    from difflet.backends.neuron.flux.transformer import flux_position_ids

    g = torch.Generator().manual_seed(seed)
    txt_ids, img_ids = flux_position_ids(config)
    return dict(
        hidden_states=torch.randn(1, config.image_seq_len, int(config.in_channels), generator=g),
        encoder_hidden_states=torch.randn(1, int(config.text_seq_len), int(config.joint_attention_dim),
                                          generator=g),
        pooled_projections=torch.randn(1, int(config.pooled_projection_dim), generator=g),
        timestep=torch.tensor([0.75]),  # sigma; diffusers' Flux scales by 1000 inside
        guidance=torch.tensor([3.5]),
        img_ids=img_ids, txt_ids=txt_ids,
    )


def make_oracle(args) -> int:
    from diffusers import FluxTransformer2DModel

    from difflet.backends.neuron.flux.config import NeuronFluxConfig

    path = os.path.join(args.model_dir, "transformer")
    config = NeuronFluxConfig.from_pretrained(path, height=args.height, width=args.width)
    inputs = _inputs(config)
    out = {"inputs": inputs, "height": args.height, "width": args.width}
    for name, dtype in (("ref_fp32", torch.float32), ("control_bf16", torch.bfloat16)):
        t = time.perf_counter()
        model = FluxTransformer2DModel.from_pretrained(path, torch_dtype=dtype).eval()
        cast = {k: (v.to(dtype) if v.is_floating_point() and k not in ("img_ids", "txt_ids") else v)
                for k, v in inputs.items()}
        with torch.no_grad():
            out[name] = model(**cast, return_dict=False)[0].float()
        del model
        print(f"{name}: {time.perf_counter() - t:.0f} s, |out| mean {out[name].abs().mean():.4f}", flush=True)
    torch.save(out, os.path.expanduser(args.make_oracle))
    return 0


def check(args) -> int:
    import torch.distributed as dist

    from difflet.backends.neuron.flux.config import NeuronFluxConfig
    from difflet.backends.neuron.flux.transformer import NeuronFluxTransformerApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    from difflet.backends.registry import get_backend

    oracle = torch.load(os.path.expanduser(args.oracle))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)  # process group + this rank's core
    path = os.path.join(args.model_dir, "transformer")
    config = NeuronFluxConfig.from_pretrained(path, height=oracle["height"], width=oracle["width"],
                                              tp_degree=world)
    app = NeuronFluxTransformerApplication(model_path=path, config=config, parallel=parallel,
                                           dtype=torch.bfloat16, exec_mode="compile")
    t = time.perf_counter()
    app.load()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"load+warmup {time.perf_counter() - t:.1f} s phases={app.phase_seconds} "
              f"report={app.load_report}", flush=True)

    def forward():
        out = app(**oracle["inputs"])
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.neuron.synchronize()
        return out

    out = forward().float().cpu()
    if rank == 0:
        ref, ctl = oracle["ref_fp32"], oracle["control_bf16"]
        rel = lambda a: ((a - ref).abs().mean() / ref.abs().mean()).item()  # noqa: E731
        ours, control = rel(out), rel(ctl)
        cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), 0).item()
        ok = torch.isfinite(out).all().item() and ours <= GATE * control
        print(f"{'PASS' if ok else 'FAIL'} parity cos={cos:.6f} rel_l1={ours:.3e} "
              f"({ours / control:.3f}x control {control:.3e}, gate {GATE}x)", flush=True)
    times = []
    for _ in range(args.iters):
        t = time.perf_counter()
        forward()
        times.append(time.perf_counter() - t)
    if rank == 0 and times:
        times.sort()
        print(f"forward median {statistics.median(times) * 1e3:.1f} ms  min {times[0] * 1e3:.1f}  "
              f"max {times[-1] * 1e3:.1f}  (n={len(times)})", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="the FLUX.1-dev snapshot directory")
    p.add_argument("--make-oracle", help="phase 1: write the oracle here (CPU only)")
    p.add_argument("--oracle", help="phase 2: the oracle to check against (under torchrun)")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    if args.make_oracle:
        return make_oracle(args)
    if args.oracle:
        return check(args)
    p.error("give --make-oracle or --oracle")


if __name__ == "__main__":
    sys.exit(main())
