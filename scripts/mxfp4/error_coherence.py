#!/usr/bin/env python3
"""Why a single forward does not rank the 50-step image: is a configuration's DiT output error
the same vector at every denoise step (it adds up over the steps) or a new one each step (it
averages out)? Replays the fp32 arm's captures as ``sensitivity.py`` does and reports, per
configuration, the error's rel L2 at every captured step and the cosine between the error
vectors of every pair of steps.

Configurations (the model's default target set unless a text-stream one):
    act8        MXFP8 activations, fp32 weights
    w8, w4      MXFP8 / FP4 (ocp) weights, fp32 activations
    w4txt       FP4 (mse) weights on the text-stream targets, everything else fp32
    mxfp8, fp4  the 50-step arms mxfp8 and fp4-ocp: both operands quantized
    txt4        the mixed arm: MXFP8 everywhere, FP4 (mse) weights on the text-stream targets

    taskset -c <8 cores> env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \\
        PYTHONPATH=<repo> python scripts/mxfp4/error_coherence.py --model qwen_image \\
        --capture-dir <out>/qwen_image/fp32 --configs act8,w8,w4 --out <out>/qwen_image/coh_a.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
    raise SystemExit("set TORCH_DEVICE_BACKEND_AUTOLOAD=0: this run is CPU only, and importing "
                     "torch otherwise loads torch_neuronx, which opens a Neuron device")

import torch
import torch.nn.functional as F

from difflet.quant.fake_linear import FakeQuantMXLinear
from difflet.quant.targets import default_spec
from scripts.fp32_ref._common import require_fp32_math, set_threads
from scripts.mxfp4.arms import _originals, apply_arm, restore, snapshot_linears
from scripts.mxfp4.sensitivity import GROUPS, load_steps

#: config -> (arm, text-stream targets only, text-stream FP4 only, activations quantized,
#: weights quantized)
CONFIGS = {
    "act8": ("mxfp8", False, False, True, False),
    "w8": ("mxfp8", False, False, False, True),
    "w4": ("fp4-ocp", False, False, False, True),
    "w4txt": ("fp4-mse", True, False, False, True),
    "mxfp8": ("mxfp8", False, False, True, True),
    "fp4": ("fp4-ocp", False, False, True, True),
    "txt4": ("fp4-mse", False, True, True, True),
}


def _fp32_activations(self: FakeQuantMXLinear, x: torch.Tensor) -> torch.Tensor:
    bias = None if self.bias is None else self.bias.float()
    return F.linear(x.float(), self.weight_dequant, bias).to(x.dtype)


def errors(dit: torch.nn.Module, model_key: str, config: str, steps: dict[str, dict]) -> dict:
    arm, txt_targets, txt_fp4, act, weights = CONFIGS[config]
    text = [p for g, p in GROUPS[model_key].items() if "txt" in g]
    saved = snapshot_linears(dit)
    originals = _originals(dit, default_spec(model_key, tp_degree=4))
    forward = FakeQuantMXLinear.forward
    try:
        apply_arm(dit, model_key, arm, targets=text if txt_targets else None,
                  fp4_targets=text if txt_fp4 else None)
        if not weights:  # the layers keep their MXFP8 activation law over the fp32 weight
            for path, weight in originals.values():
                dit.get_submodule(path).weight_dequant = weight.float()
        if not act:
            FakeQuantMXLinear.forward = _fp32_activations
        errs, rel = [], {}
        for name, step in steps.items():
            want = step["output"].double().flatten()
            errs.append(dit(**step["kwargs"])[0].double().flatten() - want)
            rel[name] = float(errs[-1].norm() / want.norm())
    finally:
        FakeQuantMXLinear.forward = forward
        restore(dit, saved)
    cos = [[float(F.cosine_similarity(a, b, dim=0)) for b in errs] for a in errs]
    off = [cos[i][j] for i in range(len(errs)) for j in range(i)]
    return {"rel_l2": rel, "cosine": cos, "mean_pair_cosine": sum(off) / len(off)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(GROUPS))
    p.add_argument("--capture-dir", required=True, type=Path, help="the fp32 arm's --out-dir")
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    configs = args.configs.split(",")
    if unknown := [c for c in configs if c not in CONFIGS]:
        p.error(f"unknown configs {unknown}: choose from {sorted(CONFIGS)}")
    torch.set_grad_enabled(False)
    require_fp32_math()
    set_threads(args.threads)
    arm = json.loads((args.capture_dir / "arm.json").read_text())
    if (arm["model"], arm["arm"]) != (args.model, "fp32"):
        p.error(f"{args.capture_dir} is a {arm['model']} {arm['arm']} run, not the fp32 arm")
    if not (args.capture_dir / "run.json").exists():  # written last: every capture is there
        p.error(f"{args.capture_dir} has no run.json: the fp32 arm has not finished")
    steps = load_steps(args.capture_dir)
    ref = importlib.import_module(f"scripts.fp32_ref.{args.model}")
    from diffusers import FluxTransformer2DModel, QwenImageTransformer2DModel

    loader = QwenImageTransformer2DModel if args.model == "qwen_image" else FluxTransformer2DModel
    dit = loader.from_pretrained(ref.SNAPSHOT, subfolder="transformer", torch_dtype=torch.float32)
    ref.require_fp32({"transformer": dit})
    found = {}
    for config in configs:
        t = time.perf_counter()
        found[config] = errors(dit, args.model, config, steps)
        args.out.write_text(json.dumps({"model": args.model, "steps": list(steps),
                                        "configs": found}, indent=1) + "\n")
        r = found[config]
        print(f"[coherence] {config}: rel L2 "
              + " ".join(f"{v * 100:.2f}%" for v in r["rel_l2"].values())
              + f", mean cosine between steps {r['mean_pair_cosine']:.3f}, "
              f"{time.perf_counter() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
