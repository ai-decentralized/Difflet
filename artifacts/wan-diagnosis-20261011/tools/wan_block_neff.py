#!/usr/bin/env python
"""Compile a reduced-depth Wan 2.1 T2V-14B backbone NEFF at a given shape, for
device profiling (neuron-explorer) of per-layer behaviour at the full sequence
length. Diagnosis only: weights are not loaded; the profiler feeds its own
inputs.

    python wan_block_neff.py --layers 2 --num-frames 81 --tag l2f81
    python wan_block_neff.py --layers 2 --num-frames 81 --sp --tag l2f81sp

Prints the compiled NEFF path (the newest MODULE_* in the neuron compile cache
produced by this run).
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import torch

CACHE = "/var/tmp/neuron-compile-cache"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    p.add_argument("--revision", default="38ec498cb3208fb688890f8cc7e94ede2cbd7f68")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--sp", action="store_true")
    p.add_argument("--tag", required=True)
    p.add_argument("--extra-cc", default="", help="extra neuronx-cc flags appended to the backbone's")
    p.add_argument("--strided-dma", action="store_true", help="add --vectorize-strided-dma to the tensorizer options")
    p.add_argument("--work-dir", default=os.path.expanduser("~/work/scratch/wanblock"))
    args = p.parse_args()

    from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication
    from difflet.models.wan.application import create_wan_backbone_config
    from difflet.pipeline.path_resolver import resolve_model_path

    base = NeuronWanBackboneApplication.get_compiler_args

    def compiler_args(self):
        s = base(self)
        if args.strided_dma:
            s = s.replace("--enable-ccop-compute-overlap'", "--enable-ccop-compute-overlap --vectorize-strided-dma'")
            assert "--vectorize-strided-dma" in s, s
        return f"{s} {args.extra_cc}".strip()

    NeuronWanBackboneApplication.get_compiler_args = compiler_args

    model_dir = resolve_model_path(args.model, revision=args.revision, local_files_only=True)
    latent_frames = (args.num_frames - 1) // 4 + 1
    config = create_wan_backbone_config(
        model_path=model_dir, world_size=args.tp, tp_degree=args.tp, dtype=torch.bfloat16,
        height=args.height, width=args.width, num_frames=latent_frames, sp_enabled=args.sp,
    )
    config.num_layers = args.layers
    out_dir = os.path.join(args.work_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    app = NeuronWanBackboneApplication(model_path=os.path.join(model_dir, "transformer"), config=config)
    t0 = time.time()
    print(f"[wanblock] layers={args.layers} tp={args.tp} sp={args.sp} "
          f"shape={args.height}x{args.width}x{args.num_frames} (latent {latent_frames}) -> {out_dir}", flush=True)
    app.compile(out_dir, debug=False)
    neffs = [n for n in glob.glob(f"{CACHE}/*/MODULE_*/model.neff") if os.path.getmtime(n) >= t0]
    print(f"[wanblock] compiled in {time.time() - t0:.0f}s", flush=True)
    for n in sorted(neffs, key=os.path.getmtime):
        print(f"[wanblock] NEFF {n} ({os.path.getsize(n) / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
