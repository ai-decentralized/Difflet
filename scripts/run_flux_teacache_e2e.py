#!/usr/bin/env python3
"""cclog 85: Flux TeaCache fused-A — compile + signal gate + calibrate + e2e A/B.

Single process (one compile, reused for baseline/teacache):
  1. DiffletPipeline.from_pretrained(FLUX.1-dev, application_kwargs={teacache_fused:True})
     -> compiles CLIP/T5/transformer/VAE + the fused probe NEFF, loads.
  2. SIGNAL GATE (cclog 84/85): record-only run (probe every step, never skip);
     Pearson(rel_l1 block0 mod, rel_l1 noise_pred). Strong (>=0.5) -> fit poly +
     accumulator; weak -> fixed cadence (probe-free).
  3. e2e A/B: baseline (controller off) vs teacache (controller on), same loop
     (_call_with_teacache), same seed -> final-latent cosine + wall-clock speedup.
Writes cclogs/m9-teacache/flux_teacache_e2e.json + the calibration JSON.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODEL = "black-forest-labs/FLUX.1-dev"
CCLOG = ROOT / "cclogs" / "m9-teacache"
CCLOG.mkdir(parents=True, exist_ok=True)
PROMPTS = [
    "a red fox sitting in a snowy forest at dawn, sharp detail",
    "a bustling night market with neon signs and steam, cinematic",
    "a close-up portrait of an old fisherman, weathered skin, soft light",
    "a glass of orange juice on a wooden table by a sunny window",
]
NUM_STEPS = 28
HEIGHT = WIDTH = 1024
# TARGET_SKIP: fraction of denoising steps to skip (0.0 = none, 0.5 = half).
# At 0.4 → 1.67x theoretical speedup; at 0.5 → 2.0x.
# Tune upward until final_cosine drops below 0.98.
TARGET_SKIP = 0.5
# ONLINE_DELTA_ALPHA: if > 0, enables cclog 91 probe-free online-delta mode.
# The controller skips a step iff the previous full step's output rel-L1 delta
# was below alpha * baseline. No per-model calibration needed.
# Set > 0 to bypass signal-gate + poly-fit; set 0 to use calibration-based mode.
ONLINE_DELTA_ALPHA = 0.0
SEED = 0


def _cos(a, b):
    return float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1), dim=1).item())


def _pearson(xs, ys):
    a = torch.tensor(xs, dtype=torch.float64)
    b = torch.tensor(ys, dtype=torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def _gen():
    return torch.Generator().manual_seed(SEED)


def _run(pipe, prompt):
    out = pipe(
        prompt=prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=3.5,
        output_type="latent",
        generator=_gen(),
        return_dict=False,
    )
    latents = out[0] if isinstance(out, (tuple, list)) else out
    return latents.detach().cpu()


def main() -> int:
    from difflet import DiffletParallelConfig, DiffletPipeline
    from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController

    print("[flux-tc] compiling + loading (teacache_fused=True)...", flush=True)
    t0 = time.time()
    pipe = DiffletPipeline.from_pretrained(
        MODEL,
        model_type="flux",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype=torch.bfloat16,
        height=HEIGHT,
        width=WIDTH,
        skip_warmup=True,
        application_kwargs={"teacache_fused": True},
    )
    print(f"[flux-tc] ready in {time.time()-t0:.0f}s; compiled_path={pipe.compiled_path}", flush=True)
    fp = pipe.app.pipe  # NeuronFluxPipeline

    # ---- signal gate (record-only) ----
    fp.teacache_controller = None
    fp._tc_record = True
    pairs = []
    for p in PROMPTS[:3]:
        _ = _run(pipe, p)
        pairs.extend(fp._tc_pairs)
        print(f"[flux-tc] gate prompt done, {len(fp._tc_pairs)} pairs", flush=True)
    fp._tc_record = False
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    pearson = _pearson(xs, ys)
    print(f"[flux-tc] SIGNAL n={len(pairs)} Pearson(rel_l1_mod, rel_l1_noise)={pearson:.4f} "
          f"| rel_l1_mod mean={np.mean(xs):.4f} range[{min(xs):.4f},{max(xs):.4f}]", flush=True)

    shape_label = f"{HEIGHT}x{WIDTH}"
    warmup, cooldown = 3, 3
    # cclog 91 online-delta mode: probe-free, calibration-free.
    # Uses the real noise_pred trajectory to detect flat steps — no per-model
    # calibration JSON needed. Set ONLINE_DELTA_ALPHA > 0 at the top of this file.
    if ONLINE_DELTA_ALPHA > 0:
        mode = "online_delta"
        calib = TeaCacheCalibration(
            model="flux", shape_label=shape_label, num_steps=NUM_STEPS,
            poly_coef=(0.0,), threshold=0.0, warmup_steps=warmup, cooldown_steps=cooldown,
            online_delta_alpha=float(ONLINE_DELTA_ALPHA),
            target_speedup=1.0 / (1.0 - TARGET_SKIP),
            fit_r2=0.0, n_samples=0,
        )
        print(f"[flux-tc] mode=online_delta alpha={ONLINE_DELTA_ALPHA} (no calibration needed)", flush=True)
    elif pearson >= 0.5:
        mode = "controller"
        coef = np.polyfit(np.array(xs), np.array(ys), 4)  # highest-degree first
        coef_asc = tuple(float(c) for c in coef[::-1])
        # tune accumulator threshold to ~TARGET_SKIP on the gate series
        poly = np.poly1d(coef)
        target_skips = max(int(round(TARGET_SKIP * NUM_STEPS)), 1)
        # simulate accumulator over a representative single-prompt series; pick
        # the threshold whose skip count is closest to target (skips is monotone
        # increasing in threshold, so this is well-defined).
        per = len(pairs) // 3 if len(pairs) >= 3 else len(pairs)
        series = xs[:per] if per else xs

        def _sim_skips(thr):
            accum = 0.0
            skips = 0
            for j, x in enumerate(series):
                if j < warmup or j >= len(series) - cooldown:
                    accum = 0.0
                    continue
                accum += abs(float(poly(x)))
                if accum < thr:
                    skips += 1
                else:
                    accum = 0.0
            return skips

        best_thr, best_err = 0.15, 1e9
        for thr in np.linspace(0.01, 2.0, 200):
            err = abs(_sim_skips(float(thr)) - target_skips)
            if err < best_err:
                best_err, best_thr = err, float(thr)
        calib = TeaCacheCalibration(
            model="flux", shape_label=shape_label, num_steps=NUM_STEPS,
            poly_coef=coef_asc, threshold=best_thr, warmup_steps=warmup,
            cooldown_steps=cooldown, accumulate=True, target_speedup=1.0 / (1.0 - TARGET_SKIP),
            fit_r2=float(pearson ** 2), n_samples=len(pairs),
        )
        print(f"[flux-tc] mode=controller poly(asc)={coef_asc} thr={best_thr:.3f}", flush=True)
    else:
        mode = "cadence"
        calib = TeaCacheCalibration(
            model="flux", shape_label=shape_label, num_steps=NUM_STEPS,
            poly_coef=(0.0,), threshold=0.0, warmup_steps=warmup, cooldown_steps=cooldown,
            cadence=2, target_speedup=1.0 / (1.0 - TARGET_SKIP), fit_r2=float(pearson ** 2),
            n_samples=len(pairs),
        )
        print("[flux-tc] mode=cadence (weak signal -> probe-free fixed cadence)", flush=True)
    calib_doc = calib.to_dict()
    calib_doc["hardware_measured"] = True
    calib_path = CCLOG / f"calibration_flux_{HEIGHT}_{NUM_STEPS}step.json"
    calib_path.write_text(json.dumps(calib_doc, indent=2, sort_keys=True) + "\n")

    # ---- baseline ----
    fp.teacache_controller = None
    base, base_t = [], []
    for p in PROMPTS:
        t = time.time()
        base.append(_run(pipe, p))
        base_t.append(time.time() - t)
        print(f"[flux-tc] baseline: {base_t[-1]:.2f}s", flush=True)

    # ---- teacache ----
    fp.teacache_controller = TeaCacheController(calib)
    fp.teacache_speedup = calib.target_speedup
    tc, tc_t, cos, skips, full = [], [], [], 0, 0
    for i, p in enumerate(PROMPTS):
        fp.teacache_controller.reset()
        t = time.time()
        lat = _run(pipe, p)
        tc_t.append(time.time() - t)
        tc.append(lat)
        cos.append(_cos(lat, base[i]))
        st = fp.teacache_controller.stats()
        skips += int(st["skipped_steps"]); full += int(st["full_steps"])
        print(f"[flux-tc] teacache: {tc_t[-1]:.2f}s cos={cos[-1]:.4f} "
              f"skip={st['skipped_steps']} full={st['full_steps']}", flush=True)

    speedup = sum(base_t) / sum(tc_t)
    final_cos = min(cos)
    result = {
        "schema": "difflet-m9-teacache-flux-e2e-v1",
        "model": "flux", "shape_label": shape_label, "num_steps": NUM_STEPS,
        "n_prompts": len(PROMPTS), "signal_pearson": pearson, "mode": mode,
        "baseline_total_s": sum(base_t), "teacache_total_s": sum(tc_t),
        "measured_speedup": speedup, "final_cosine": final_cos,
        "skipped_steps": skips, "full_steps": full, "hardware_measured": True,
    }
    (CCLOG / "flux_teacache_e2e.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\n[flux-tc] ===== RESULT =====", flush=True)
    print(f"[flux-tc] mode={mode} signal Pearson={pearson:.4f}", flush=True)
    print(f"[flux-tc] SPEEDUP={speedup:.3f}x  final_cosine={final_cos:.4f}  "
          f"skipped={skips}/{skips+full}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
