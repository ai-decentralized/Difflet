#!/usr/bin/env python3
"""DP correctness: dp=k outputs must be bit-identical to dp=1 (same requests).

Usage (on trn2, weights + compile cache present — detach long runs):
  setsid python scripts/verify_dp_correctness.py \
      --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --dp 2 --tp-degree 4 \
      --steps 2 > /tmp/dp_correctness.log 2>&1 &

Evidence to check in the log beyond exit code (repo rule: exit-zero is not
verification): the [dp-router] core-pinning lines, and ZERO recompilation in
the dp=k run (compile-once-load-k cache reuse).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

PROMPTS = ["a red fox in snow", "a sailboat at dusk", "a neon city street",
           "a bowl of ramen"]


def run_batch(model_id, dp, tp, steps, out_dir, extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    req_file = out_dir / "requests.jsonl"
    lines = [
        {"prompt": p, "output": str(out_dir / f"out_{i}.mp4"), "seed": 1000 + i}
        for i, p in enumerate(PROMPTS)
    ]
    req_file.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    cmd = [sys.executable, "-m", "difflet.cli.main", "generate",
           "--model-id", model_id, "--dp", str(dp), "--tp-degree", str(tp),
           "--steps", str(steps), "--requests", str(req_file),
           "--work-dir", str(out_dir / "work"), "--keep-work-dir", *extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    # Compare pre-VAE latents from the kept work dirs (spec §Testing 1): final
    # outputs may be mp4 (encoder is not bit-stable); the denoised latents are.
    latents = []
    for i in range(len(lines)):
        matches = list((out_dir / "work").rglob(f"latents_req{i:04d}.pt"))
        if len(matches) != 1:
            sys.exit(f"expected exactly one latents_req{i:04d}.pt under "
                     f"{out_dir / 'work'}, found {matches}")
        latents.append(matches[0])
    return latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--tp-degree", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("extra", nargs="*", default=[])
    args = ap.parse_args()

    base = Path(tempfile.mkdtemp(prefix="dp_correctness_"))
    serial = run_batch(args.model_id, 1, args.tp_degree, args.steps,
                       base / "serial", args.extra)
    parallel = run_batch(args.model_id, args.dp, args.tp_degree, args.steps,
                         base / "parallel", args.extra)

    failures = 0
    for s, p in zip(serial, parallel):
        a, b = torch.load(s), torch.load(p)
        if torch.equal(a, b):
            print(f"PASS bit-identical: {s.name}")
        else:
            diff = (a.float() - b.float()).abs().max().item()
            print(f"FAIL {s.name}: max_abs_diff={diff:.3e}")
            failures += 1
    print(f"[verify_dp_correctness] {len(serial) - failures}/{len(serial)} identical")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
