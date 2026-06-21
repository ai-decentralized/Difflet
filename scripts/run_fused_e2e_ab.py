#!/usr/bin/env python3
"""cclog 80 Step 5: e2e A/B — fused-A probe vs baseline, on production 50-step.

Builds the HV app with teacache_fused=True (prev_mod persistent on device,
probe returns only delta), compiles the fused probe + reuses the cached DiT,
then runs the full 50-step denoise on the 8 holdout bundles:
  - baseline: controller off
  - fused teacache: target=2.0 calibration
Reports wall-clock speedup + trajectory/final cosine. Compare against the
non-fused 1.728x recorded in cclog 77.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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

SOURCE = ROOT / ".difflet-cache" / "f3_hunyuan_n4_4d8s1r" / "source"
COMPILED = ROOT / ".difflet-cache" / "f3_hunyuan_n4_4d8s1r" / "compiled"
CALIB = ROOT / "cclogs" / "m9-teacache" / "calibration_hunyuan_video_n4_4d8s1r_50step_t2.0.json"
BUNDLE_DIR = ROOT / ".difflet-cache" / "hunyuan_dit_inputs" / "m9_calib_50step"
NUM_STEPS = 50


def main() -> int:
    from difflet.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController
    from scripts.verify_teacache_speedup import (
        _build_hv_bundle,
        _load_hv_bundle,
        _run_hv_bundle,
        _tensor_cosine,
        _trajectory_cosine,
    )

    holdout = sorted(BUNDLE_DIR.glob("holdout_*.safetensors"))
    records = [_load_hv_bundle(p) for p in holdout]
    first_meta = records[0][0]

    app = NeuronHunyuanVideoApplication(
        model_path=str(SOURCE),
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype=torch.bfloat16,
        shape={"height": int(first_meta["height"]), "width": int(first_meta["width"]),
               "num_frames": int(first_meta["num_frames"])},
        text_seq_len=int(first_meta["text_seq_len"]),
        enable_vae_decoder=False,
        teacache_fused=True,
    )
    print(f"[e2e] components: {[c.name for c in app.components()]}", flush=True)
    # Compile ONLY the fused probe (DiT stays cached at compiled/transformer/).
    print("[e2e] compiling fused probe only...", flush=True)
    t = time.perf_counter()
    app.teacache_probe.compile(str(COMPILED / "teacache_probe"))
    print(f"[e2e] fused probe compiled in {time.perf_counter() - t:.1f}s", flush=True)
    app.load(str(COMPILED), skip_warmup=True)
    print("[e2e] loaded", flush=True)

    # baseline (no teacache)
    app.pipeline.teacache_controller = None
    base = []
    for meta, tensors in records:
        out, el = _run_hv_bundle(app, meta=meta, tensors=tensors, num_steps=NUM_STEPS)
        base.append((out, el))
    base_total = sum(e for _, e in base)

    # fused teacache target=2.0
    calib = TeaCacheCalibration.from_json(str(CALIB))
    app.pipeline.teacache_controller = TeaCacheController(calib)
    app.pipeline.teacache_speedup = 2.0
    fused_times, tcos, fcos, skips, full = [], [], [], 0, 0
    for i, (meta, tensors) in enumerate(records):
        app.pipeline.teacache_controller.reset()
        out, el = _run_hv_bundle(app, meta=meta, tensors=tensors, num_steps=NUM_STEPS)
        bout, _ = base[i]
        fused_times.append(el)
        tcos.append(_trajectory_cosine(out.trajectory, bout.trajectory))
        fcos.append(_tensor_cosine(out.latents, bout.latents))
        st = app.pipeline.teacache_controller.stats()
        skips += int(st["skipped_steps"]); full += int(st["full_steps"])
        print(f"[e2e] bundle {i+1}: fused={el:.3f}s baseline={base[i][1]:.3f}s", flush=True)
    fused_total = sum(fused_times)

    result = {
        "schema": "difflet-m9-teacache-fused-e2e-v1",
        "num_steps": NUM_STEPS, "n_bundles": len(records),
        "baseline_total_s": base_total, "fused_total_s": fused_total,
        "measured_speedup": base_total / fused_total,
        "trajectory_cosine": min(tcos), "final_cosine": min(fcos),
        "skipped_steps": skips, "full_steps": full,
        "nonfused_speedup_ref": 1.728, "hardware_measured": True,
    }
    out_path = ROOT / "cclogs" / "m9-teacache" / "fused_e2e_ab.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[e2e] SPEEDUP fused={result['measured_speedup']:.4f}x "
          f"(non-fused was 1.728x) | traj={result['trajectory_cosine']:.6f} "
          f"final={result['final_cosine']:.6f}", flush=True)
    print(f"[e2e] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
