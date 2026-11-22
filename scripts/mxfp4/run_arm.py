#!/usr/bin/env python3
"""One phase-0 arm (fp32 | mxfp8 | fp4-ocp | fp4-mse) through ``scripts/fp32_ref/<model>.py``'s
own main: the reference runs unchanged (its settings, the device's noise and its md5 check,
``save_run``), and the arm is put on the loaded DiT right after the reference's fp32 check.

    taskset -c <8 cores> env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \\
        PYTHONPATH=<repo> python scripts/mxfp4/run_arm.py --model qwen_image --arm fp4-ocp \\
        --out-dir <out>/qwen_image/fp4-ocp

Writes what the reference writes (latents.pt, out.png, run.json with step_s; its load_s includes
the swap) and arm.json; with --capture-steps also the DiT's keyword arguments and output at those
steps (dit_stepNN.pt), the inputs of sensitivity.py. A step is one DiT call at the reference's
settings: both pipelines call the DiT once per step (true_cfg_scale 1.0, no negative prompt), by
keyword only, with return_dict=False. --out-dir must be new or empty: one run per directory, its
log elsewhere.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
    raise SystemExit("set TORCH_DEVICE_BACKEND_AUTOLOAD=0: this run is CPU only, and importing "
                     "torch otherwise loads torch_neuronx, which opens a Neuron device")

import torch

from scripts.mxfp4.arms import ARMS, apply_arm


def capture(dit: torch.nn.Module, out_dir: Path, steps: set[int]) -> list[int]:
    """Save the DiT's forward number i, for i in ``steps``, as ``out_dir/dit_stepNN.pt``:
    ``{"kwargs", "output"}``, replayable as ``dit(**kwargs)[0]``. Returns the running call
    count, ``[n]``."""
    calls = [0]

    def hook(_module, args, kwargs, output):
        if args:
            raise RuntimeError("the DiT was called with positional arguments: not replayable")
        if calls[0] in steps:
            # Copies, so the file holds these tensors alone and nothing the pipeline does later.
            kept = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in kwargs.items()}
            torch.save({"kwargs": kept, "output": output[0].detach().clone()},
                       out_dir / f"dit_step{calls[0]:02d}.pt")
        calls[0] += 1

    dit.register_forward_hook(hook, with_kwargs=True)
    return calls


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=("qwen_image", "flux"))
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--targets", default="",
                   help="semicolon-separated regexes replacing the model's default target set")
    p.add_argument("--fp4-targets", default="",
                   help="semicolon-separated regexes of the quantized names an fp4 arm makes FP4, "
                        "the others staying MXFP8 (default: all)")
    p.add_argument("--tp", type=int, default=4,
                   help="the TP degree whose K shards the blocks follow")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--capture-steps", default="",
                   help="comma-separated 0-based steps whose DiT call to save")
    args = p.parse_args()
    targets = [t for t in args.targets.split(";") if t] or None
    fp4_targets = [t for t in args.fp4_targets.split(";") if t] or None
    steps = {int(s) for s in args.capture_steps.split(",") if s}
    if steps and not 0 <= min(steps) <= max(steps) < args.steps:
        p.error(f"--capture-steps {sorted(steps)} are not all steps of a {args.steps}-step run")
    # apply_arm refuses these two as well, but only once the pipeline has loaded
    if fp4_targets and not args.arm.startswith("fp4-"):
        p.error(f"--fp4-targets needs an fp4 arm, not {args.arm}")
    if targets and args.arm == "fp32":
        p.error("the fp32 arm quantizes nothing: it takes no --targets")
    # An earlier run's dit_stepNN.pt would be read as this run's steps, and its latents.pt and
    # out.png would sit under this run's arm.json until this run ends.
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        p.error(f"--out-dir {args.out_dir} is not empty: one run per directory")
    ref = importlib.import_module(f"scripts.fp32_ref.{args.model}")
    guard = ref.require_fp32
    calls = [0]

    def after_guard(components: dict) -> None:
        nonlocal calls
        guard(components)  # the reference's own fp32 check, on the pipeline as loaded
        dit = components["transformer"]
        names = apply_arm(dit, args.model, args.arm, targets=targets, fp4_targets=fp4_targets,
                          tp=args.tp)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "arm.json").write_text(json.dumps(
            {"model": args.model, "arm": args.arm, "tp": args.tp, "targets": targets,
             "fp4_targets": fp4_targets, **names}, indent=1) + "\n")
        if steps:
            calls = capture(dit, args.out_dir, steps)

    ref.require_fp32 = after_guard  # main() looks the name up in its module when it calls it
    status = ref.main(["--out-dir", str(args.out_dir), "--threads", str(args.threads),
                       "--steps", str(args.steps)])
    if steps and calls[0] != args.steps:
        raise RuntimeError(f"the DiT ran {calls[0]} times in {args.steps} steps: the "
                           f"dit_stepNN.pt files are not those steps")
    return status


if __name__ == "__main__":
    sys.exit(main())
