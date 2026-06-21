#!/usr/bin/env python3
"""cclog 92: Qwen-Image ONLINE-DELTA TeaCache e2e A/B (device).

The cclog-82 adaptive e2e (`run_qwen_teacache_e2e.py`) measured Qwen's block-0
probe Pearson at ~0.30 — a dead signal. cclog 92 showed Qwen's *output*
trajectory is self-predictable (δ-autocorr 0.93), so the right method is the
generic, probe-free **online_delta** controller (skip when the previous full
step's measured noise_pred rel-L1 δ < α·baseline). This script proves it on
device: same proven harness setup as the adaptive e2e (cached 20B DiT, real
50-step holdout bundles, real FlowMatch scheduler), but the controller is built
from the emitted online calib instead of a fitted poly. Zero per-model code —
the existing Qwen denoise loop already drives online mode (needs_signal()=False
→ no probe; record_full_step feeds the output δ).

Reports: baseline (cache off) vs online (cache on) wall-clock speedup +
trajectory/final cosine + skip count, written to cclogs/m9-teacache/.

Run:
  PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \
  NEURON_RT_NUM_CORES=4 PYTHONPATH=/home/ubuntu/difflet \
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python \
      scripts/run_qwen_online_delta_e2e.py --online-calib \
      cclogs/m9-teacache/teacache_calib_qwen_image_online.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Reuse the proven cclog-82 harness (setup, bundle IO, cosine, runner). Importing
# the module runs its ensure_runtime_python() (harmless when torch is present)
# but NOT main() (guarded by __main__).
from scripts.run_qwen_teacache_e2e import (  # noqa: E402
    BUNDLE_DIR,
    CCLOG,
    COMPILED,
    DEFAULT_MODEL_DIR,
    DEFAULT_TRANSFORMER_CACHE,
    MODEL,
    _load_bundle,
    _run_bundle,
    _setup_compiled_dir,
    _tensor_cosine,
    _trajectory_cosine,
)

import torch  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--online-calib",
                   default=str(Path("cclogs/m9-teacache/teacache_calib_qwen_image_online.json")),
                   help="online_delta calibration JSON (emit_online_delta_calib.py output)")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--scheduler-id", default="Qwen/Qwen-Image")
    p.add_argument("--transformer-cache", default=str(DEFAULT_TRANSFORMER_CACHE))
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--text-seq-len", type=int, default=1024)
    p.add_argument("--max-bundles", type=int, default=0, help="0 = all holdout bundles")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    from difflet.models.qwen_image.application import NeuronQwenImageApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController

    calib = TeaCacheCalibration.from_dict(json.loads(Path(args.online_calib).read_text()))
    if float(getattr(calib, "online_delta_alpha", 0.0)) <= 0.0:
        raise ValueError(
            f"{args.online_calib} is not an online_delta calib "
            f"(online_delta_alpha={getattr(calib, 'online_delta_alpha', None)})"
        )
    print(f"[qwen-online] calib: model={calib.model} shape={calib.shape_label} "
          f"alpha={calib.online_delta_alpha} cadence={calib.cadence} "
          f"warmup={calib.warmup_steps} cooldown={calib.cooldown_steps} "
          f"needs_signal={TeaCacheController(calib).needs_signal()}", flush=True)

    holdout = sorted(BUNDLE_DIR.glob("holdout_*.safetensors"))
    if not holdout:
        raise FileNotFoundError(f"no holdout bundles in {BUNDLE_DIR}")
    if args.max_bundles > 0:
        holdout = holdout[: args.max_bundles]
    print(f"[qwen-online] {len(holdout)} holdout bundles", flush=True)

    _setup_compiled_dir(Path(args.transformer_cache))
    app = NeuronQwenImageApplication(
        model_path=args.model_dir,
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width},
        text_seq_len=args.text_seq_len,
        teacache_fused=True,
    )
    print("[qwen-online] compiling fused probe (unused by online mode, keeps load path identical)...",
          flush=True)
    app.teacache_probe.compile(str(COMPILED / "teacache_probe"))
    app.load(str(COMPILED), skip_warmup=True)

    from diffusers import FlowMatchEulerDiscreteScheduler

    app.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.scheduler_id, subfolder="scheduler", local_files_only=True
    )
    print(f"[qwen-online] loaded; scheduler <- {args.scheduler_id}", flush=True)

    # --- baseline: controller off (full DiT every step) ---
    app.pipeline.teacache_controller = None
    baselines: list[tuple[Any, float]] = []
    for path in holdout:
        _, tensors = _load_bundle(path)
        out, el = _run_bundle(app, tensors, args.num_steps)
        baselines.append((out, el))
        print(f"[qwen-online] baseline {path.name}: {el:.3f}s", flush=True)

    # --- online: controller on (generic measured-δ skip) ---
    app.pipeline.teacache_controller = TeaCacheController(calib)
    app.pipeline.teacache_speedup = 0.0  # online mode ignores target_speedup
    online_times, tcos, fcos, skips, full = [], [], [], 0, 0
    for i, path in enumerate(holdout):
        _, tensors = _load_bundle(path)
        app.pipeline.teacache_controller.reset()
        out, el = _run_bundle(app, tensors, args.num_steps)
        bout, bel = baselines[i]
        online_times.append(el)
        tcos.append(_trajectory_cosine(out.trajectory, bout.trajectory))
        fcos.append(_tensor_cosine(out.latents, bout.latents))
        st = app.pipeline.teacache_controller.stats()
        skips += int(st["skipped_steps"])
        full += int(st["full_steps"])
        print(f"[qwen-online] online {path.name}: {el:.3f}s baseline={bel:.3f}s "
              f"skip={st['skipped_steps']} full={st['full_steps']} "
              f"final_cos={fcos[-1]:.6f}", flush=True)

    baseline_total = sum(e for _, e in baselines)
    online_total = sum(online_times)
    speedup = baseline_total / online_total if online_total else 0.0
    traj_cos = min(tcos) if tcos else 0.0
    final_cos = min(fcos) if fcos else 0.0

    ab = {
        "schema": "difflet-m9-teacache-online-delta-e2e-v1",
        "model": MODEL,
        "shape_label": calib.shape_label,
        "num_steps": int(args.num_steps),
        "method": "online_delta",
        "online_delta_alpha": float(calib.online_delta_alpha),
        "n_bundles": len(holdout),
        "baseline_total_s": baseline_total,
        "online_total_s": online_total,
        "measured_speedup": speedup,
        "trajectory_cosine": traj_cos,
        "final_cosine": final_cos,
        "skipped_steps": skips,
        "full_steps": full,
        "hardware_measured": True,
    }
    (CCLOG / "qwen_online_delta_e2e_ab.json").write_text(
        json.dumps(ab, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[qwen-online] SPEEDUP online_delta={speedup:.4f}x | traj={traj_cos:.6f} "
        f"final={final_cos:.6f} | skipped={skips}/{skips + full} -> qwen_online_delta_e2e_ab.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
