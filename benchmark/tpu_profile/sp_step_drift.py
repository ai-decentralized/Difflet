"""How far an SP variant's output drifts after k denoising steps (TPU).

    DIFFLET_TPU_SP=1 DIFFLET_TPU_SP_KERNEL=pallas \\
        python -m benchmark.tpu_profile.sp_step_drift --model wan_2_1 --steps 1,4 --out DIR --tag pallas

Runs main's TPU bench adapter on the matrix entry with only the step count
changed (a local copy of the config; ``benchmark/models.py`` is untouched) and
saves the first synced request's frames to ``DIR/<tag>-s<k>``. Run it once per
variant (TP, SP + JAX ring, SP + Pallas kernel; the variant comes from the
environment, as for the bench), then compare the frames with ``image_psnr``.

The question it answers (2026-11-11): after 20 steps the Pallas AG kernel's
video differs from the JAX ring's by 26 dB (Wan) / 31 dB (LTX-2) although the
op is bit-identical in isolation. If the 1-step outputs differ by about as much
as TP vs the ring, the 20-step gap is trajectory amplification of rounding
differences in the surrounding fusions; if they already differ by much more,
something in the model computes differently and needs bisecting.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="wan_2_1")
    parser.add_argument("--steps", default="1,4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=None, help="override the matrix seed")
    args = parser.parse_args()

    from benchmark.adapters.tpu import TpuAdapter
    from benchmark.models import MATRIX

    for k in (int(s) for s in args.steps.split(",")):
        cfg = dataclasses.replace(MATRIX[args.model], steps=k)
        if args.seed is not None:
            cfg = dataclasses.replace(cfg, seed=args.seed)
        adapter = TpuAdapter(save_dir=f"{args.out}/{args.tag}-s{k}")
        try:
            adapter.prepare(cfg)
            g = adapter.run_generate(cfg)
            print(f"[drift] {args.tag} steps={k}: output {g.get('output')}", flush=True)
        finally:
            adapter.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
