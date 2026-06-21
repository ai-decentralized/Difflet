#!/usr/bin/env python3
"""cclog 84 follow-up: clean SAME-PLATFORM Trainium A/B/C for Qwen TeaCache.

Resolves the cclog-84 §2 confound (CPU even-cadence 0.98 vs Trainium controller
0.896 changed BOTH platform and strategy). Here all three run in ONE Trainium
process against the SAME no-skip Trainium baseline, at matched skip:

  A baseline       — controller off, full DiT every step (the cosine denominator)
  B fixed-cadence  — skip every other step by index (probe-FREE, no signal/calibration)
  C controller     — rel-L1 + vLLM poly + accumulator (cclog 83, thr 0.15)

Decomposition:
  (B vs C) on Trainium  -> pure STRATEGY comparison (platform fixed)
  (B Trainium vs the CPU ablation's 0.98) -> pure PLATFORM check
Decision: ship probe-free fixed cadence if B >= C (simpler); keep controller if C >> B;
if B on Trainium is ~0.90 (not 0.98) the CPU 0.98 was a platform artifact -> accept ~0.90 ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import torch  # noqa: E402

from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController  # noqa: E402
from scripts.run_qwen_teacache_e2e import (  # noqa: E402
    BUNDLE_DIR,
    CCLOG,
    _load_bundle,
    _run_bundle,
    _tensor_cosine,
    _trajectory_cosine,
)
from scripts.run_qwen_teacache_relL1 import VLLM_POLY_ASCENDING, _build_and_load_app  # noqa: E402

WARMUP = 2
COOLDOWN = 2


def _fixed_cadence_calib(num_steps: int, cadence: int) -> TeaCacheCalibration:
    return TeaCacheCalibration(
        model="qwen_image", shape_label="1024x1024", num_steps=num_steps,
        poly_coef=(0.0,), threshold=0.0, warmup_steps=WARMUP, cooldown_steps=COOLDOWN,
        cadence=cadence, mod_input_source="block0_modulated_input",
    )


def _controller_calib(num_steps: int, threshold: float) -> TeaCacheCalibration:
    return TeaCacheCalibration(
        model="qwen_image", shape_label="1024x1024", num_steps=num_steps,
        poly_coef=VLLM_POLY_ASCENDING, threshold=threshold, warmup_steps=WARMUP,
        cooldown_steps=COOLDOWN, accumulate=True, mod_input_source="block0_modulated_input",
    )


def _run_strategy(app, bundles, num_steps, baselines, label, calib):
    app.pipeline.teacache_controller = TeaCacheController(calib)
    app.pipeline.teacache_speedup = None
    times, tcos, fcos, skips, full = [], [], [], 0, 0
    for i, path in enumerate(bundles):
        _, tns = _load_bundle(path)
        app.pipeline.teacache_controller.reset()
        out, el = _run_bundle(app, tns, num_steps)
        bout, _ = baselines[i]
        times.append(el)
        tcos.append(_trajectory_cosine(out.trajectory, bout.trajectory))
        fcos.append(_tensor_cosine(out.latents, bout.latents))
        st = app.pipeline.teacache_controller.stats()
        skips += int(st["skipped_steps"]); full += int(st["full_steps"])
        print(f"[ab] {label} {path.name}: {el:.2f}s skip={st['skipped_steps']} full={st['full_steps']}", flush=True)
    base_total = sum(e for _, e in baselines)
    return {
        "strategy": label, "measured_speedup": base_total / sum(times),
        "trajectory_cosine": min(tcos), "final_cosine": min(fcos),
        "skipped_steps": skips, "full_steps": full, "hardware_measured": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="/home/ubuntu/.cache/huggingface/hub/qwen-image-real")
    ap.add_argument("--scheduler-id", default="Qwen/Qwen-Image")
    ap.add_argument("--transformer-cache",
                    default=str(ROOT / ".difflet-cache" / "qwen_image_transformer_full" / "qwen_image"))
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--tp-degree", type=int, default=4)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--text-seq-len", type=int, default=1024)
    ap.add_argument("--cadence", type=int, default=2, help="fixed-cadence: skip every Nth step")
    ap.add_argument("--threshold", type=float, default=0.15, help="controller accumulator threshold")
    args = ap.parse_args()

    app = _build_and_load_app(args)
    bundles = sorted(BUNDLE_DIR.glob("holdout_*.safetensors"))
    print(f"[ab] {len(bundles)} holdout bundles", flush=True)

    # A: baseline (no skip)
    app.pipeline.teacache_controller = None
    baselines = []
    for path in bundles:
        _, tns = _load_bundle(path)
        out, el = _run_bundle(app, tns, args.num_steps)
        baselines.append((out, el))
        print(f"[ab] baseline {path.name}: {el:.2f}s", flush=True)

    # B: fixed cadence (probe-free)   C: controller (probe + accumulator)
    fixed = _run_strategy(app, bundles, args.num_steps, baselines, "fixed_cadence",
                          _fixed_cadence_calib(args.num_steps, args.cadence))
    ctrl = _run_strategy(app, bundles, args.num_steps, baselines, "controller",
                         _controller_calib(args.num_steps, args.threshold))

    doc = {
        "schema": "difflet-m9-teacache-fixed-cadence-ab-v1",
        "model": "qwen_image", "shape_label": "1024x1024", "num_steps": args.num_steps,
        "n_bundles": len(bundles), "warmup": WARMUP, "cooldown": COOLDOWN,
        "cadence": args.cadence, "controller_threshold": args.threshold,
        "baseline_total_s": sum(e for _, e in baselines),
        "fixed_cadence": fixed, "controller": ctrl,
        "cpu_ablation_even_cadence_final_cos": 0.9815,  # cclog 84 §1, for the platform check
        "hardware_measured": True,
    }
    (CCLOG / "qwen_fixed_cadence_ab.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print("\n[ab] ===== RESULT (Trainium, same baseline) =====", flush=True)
    print(f"[ab] fixed-cadence : {fixed['measured_speedup']:.3f}x  final={fixed['final_cosine']:.4f}  "
          f"traj={fixed['trajectory_cosine']:.4f}  skip={fixed['skipped_steps']}/{fixed['skipped_steps']+fixed['full_steps']}", flush=True)
    print(f"[ab] controller    : {ctrl['measured_speedup']:.3f}x  final={ctrl['final_cosine']:.4f}  "
          f"traj={ctrl['trajectory_cosine']:.4f}  skip={ctrl['skipped_steps']}/{ctrl['skipped_steps']+ctrl['full_steps']}", flush=True)
    d_strategy = fixed["final_cosine"] - ctrl["final_cosine"]
    d_platform = fixed["final_cosine"] - 0.9815
    print(f"[ab] STRATEGY (fixed - controller, final cos) = {d_strategy:+.4f}", flush=True)
    print(f"[ab] PLATFORM (Trainium fixed - CPU 0.9815)   = {d_platform:+.4f}", flush=True)
    if abs(d_strategy) <= 0.005:
        v = "fixed ~= controller -> SHIP probe-free fixed cadence (simpler)"
    elif d_strategy > 0.005:
        v = "fixed > controller -> fixed cadence wins, drop the probe"
    else:
        v = "controller > fixed -> signal helps, keep the controller"
    print(f"[ab] VERDICT(strategy): {v}", flush=True)
    print(f"[ab] PLATFORM: Trainium fixed-cadence final={fixed['final_cosine']:.4f} "
          f"({'matches' if d_platform > -0.02 else 'BELOW'} CPU 0.9815)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
