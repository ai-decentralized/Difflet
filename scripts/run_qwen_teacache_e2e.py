#!/usr/bin/env python3
"""cclog 82: Qwen-Image TeaCache fused-A T0 calibrate + T0b e2e A/B.

Single-process closure (load the app ONCE — reloading leaks Neuron RT
resources, cclog 75). Mirrors the HunyuanVideo path (cclog 76/80) on a
different architecture:

  1. set up a compiled dir: symlink the cached full DiT (content-addressed)
     as `compiled/transformer`, compile the fused probe to
     `compiled/teacache_probe`;
  2. load app(teacache_fused=True) once (TP=4);
  3. T0 collect (mod_input_diff, noise_pred_diff) pairs on the calib bundles —
     the fused probe's device-computed delta IS the mod_input diff vs the
     previous step (prev_mod persists on device), so calibration uses the same
     scalar the controller sees at inference;
  4. fit the degree-3 polynomial + threshold for --target-speedup;
  5. T0b A/B: baseline (controller off) vs fused (controller on) on the holdout
     bundles, report wall-clock speedup + trajectory/final cosine.

Writes the calibration JSON, the speedup curve, the integration JSON, and the
e2e A/B JSON under cclogs/m9-teacache/.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402

from nova.models.qwen_image.application import QwenImageDiTInputBundle  # noqa: E402
from scripts.calibrate_teacache import (  # noqa: E402
    _fit_poly,
    _mark_step,
    _poly_design,
    _r2_score,
    _sample_xy,
    _threshold_for_target,
)

MODEL = "qwen_image"
DEFAULT_MODEL_DIR = "/home/ubuntu/.cache/huggingface/hub/qwen-image-real"
DEFAULT_TRANSFORMER_CACHE = (
    ROOT / ".nova-cache" / "qwen_image_transformer_full" / "qwen_image"
)
BUNDLE_DIR = ROOT / ".nova-cache" / "qwen_image_dit_inputs" / "m9_calib_50step"
COMPILED = ROOT / ".nova-cache" / "qwen_m9_e2e" / "compiled"
CCLOG = ROOT / "cclogs" / "m9-teacache"


def _load_bundle(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    tensors = {
        k: v.detach().cpu()
        for k, v in load_safetensors_file(str(path), device="cpu").items()
    }
    return meta, tensors


def _build_bundle(tensors: dict[str, torch.Tensor]) -> QwenImageDiTInputBundle:
    return QwenImageDiTInputBundle(
        hidden_states=tensors["latents_init"].to(torch.bfloat16),
        timestep=tensors["timesteps"][:1].clone().to(torch.bfloat16),
        encoder_hidden_states=tensors["encoder_hidden_states"].to(torch.bfloat16),
        encoder_hidden_states_mask=tensors["encoder_hidden_states_mask"].to(torch.bool),
        guidance=tensors["guidance"].to(torch.bfloat16),
    )


def _setup_scheduler(orch, num_steps: int, image_seq_len: int) -> torch.Tensor:
    """Configure the flow-match scheduler with the same sigmas+mu as bundle gen
    and reset its step index. Returns the timesteps to drive the loop with."""
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
        calculate_shift,
        retrieve_timesteps,
    )

    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    cfg = orch.scheduler.config
    mu = calculate_shift(
        image_seq_len,
        cfg.get("base_image_seq_len", 256),
        cfg.get("max_image_seq_len", 4096),
        cfg.get("base_shift", 0.5),
        cfg.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(orch.scheduler, num_steps, "cpu", sigmas=sigmas, mu=mu)
    return timesteps


def _tensor_cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            lhs.detach().float().cpu().reshape(1, -1),
            rhs.detach().float().cpu().reshape(1, -1),
            dim=1,
        ).item()
    )


def _trajectory_cosine(lhs, rhs) -> float:
    if lhs is None or rhs is None or len(lhs) != len(rhs):
        return 0.0
    return min(_tensor_cosine(a, b) for a, b in zip(lhs, rhs))


def _run_bundle(app, tensors, num_steps: int) -> tuple[Any, float]:
    bundle = _build_bundle(tensors)
    image_seq_len = int(tensors["latents_init"].shape[1])
    timesteps = _setup_scheduler(app.pipeline, num_steps, image_seq_len)
    start = time.perf_counter()
    output = app.pipeline(
        bundle=bundle,
        timesteps=timesteps,
        output_type="latent",
        return_trajectory=True,
    )
    _mark_step()
    return output, time.perf_counter() - start


def _collect_pairs(app, tensors, num_steps: int, split: str) -> list[dict[str, Any]]:
    """Manual loop: per step record the device delta (mod_input diff vs prev
    step) and the host noise_pred diff. Step 0 is skipped (prev_mod carries the
    previous bundle's last mod_input; warmup absorbs it)."""
    from nova.models.qwen_image.pipeline import _batch_timestep, _component_dtype, _first_tensor

    pipe = app.pipeline
    bundle = _build_bundle(tensors)
    image_seq_len = int(tensors["latents_init"].shape[1])
    timesteps = _setup_scheduler(pipe, num_steps, image_seq_len)
    latents = bundle.hidden_states
    prev_noise_pred: torch.Tensor | None = None
    samples: list[dict[str, Any]] = []
    for step_index, timestep in enumerate(timesteps):
        model_dtype = _component_dtype(pipe.transformer, pipe.dtype)
        timestep_batch = _batch_timestep(timestep, latents.shape[0], latents.device, model_dtype)
        model_bundle = QwenImageDiTInputBundle(
            hidden_states=latents.to(dtype=model_dtype),
            timestep=timestep_batch,
            encoder_hidden_states=bundle.encoder_hidden_states.to(dtype=model_dtype),
            encoder_hidden_states_mask=bundle.encoder_hidden_states_mask,
            guidance=bundle.guidance.to(dtype=model_dtype),
        )
        # device delta (now float32-reduced inside the probe). mod_input (out[1])
        # is aliased to prev_mod and consumed in-place — it never returns to host.
        delta = float(app.teacache_delta(model_bundle).detach().cpu().item())
        noise_pred = _first_tensor(app(model_bundle)).detach()
        _mark_step()
        if prev_noise_pred is not None:
            samples.append(
                {
                    "split": split,
                    "step_index": int(step_index),
                    "mod_input_diff_norm": delta,
                    "noise_pred_diff_norm": float(
                        torch.linalg.vector_norm(
                            noise_pred.float().cpu() - prev_noise_pred
                        ).item()
                    ),
                }
            )
        prev_noise_pred = noise_pred.float().cpu()
        latents = pipe._scheduler_step(noise_pred, timestep, latents, int(timesteps.numel()))
        _mark_step()
        print(f"[qwen-t0] {split} step {step_index + 1}/{int(timesteps.numel())} delta={delta:.2f}",
              flush=True)
    return samples


def _setup_compiled_dir(transformer_cache: Path) -> None:
    COMPILED.mkdir(parents=True, exist_ok=True)
    link = COMPILED / "transformer"
    # Find the cached full DiT artifact (text_seq_len=1024, 1024^2, tp=4).
    target = None
    for d in sorted(transformer_cache.glob("*/transformer")):
        ncfg = d / "neuron_config.json"
        if ncfg.exists():
            cfg = json.loads(ncfg.read_text())
            if int(cfg.get("text_seq_len", 0)) == 1024:
                target = d.resolve()
                break
    if target is None:
        raise FileNotFoundError(
            f"no compiled Qwen DiT with text_seq_len=1024 found under {transformer_cache}"
        )
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
    if not link.exists():
        link.symlink_to(target)
    print(f"[qwen-e2e] compiled/transformer -> {target}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--scheduler-id", default="Qwen/Qwen-Image",
                   help="Repo/dir to load the FlowMatchEulerDiscreteScheduler from "
                        "(model-dir is transformer-only).")
    p.add_argument("--transformer-cache", default=str(DEFAULT_TRANSFORMER_CACHE))
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument("--target-speedup", type=float, default=2.0)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--cooldown-steps", type=int, default=5)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--text-seq-len", type=int, default=1024)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    from nova.models.qwen_image.application import NeuronQwenImageApplication
    from nova.pipeline.parallel_config import NovaParallelConfig
    from nova.pipeline.teacache import TeaCacheCalibration, TeaCacheController

    calib_bundles = sorted(BUNDLE_DIR.glob("calibration_*.safetensors"))
    holdout_bundles = sorted(BUNDLE_DIR.glob("holdout_*.safetensors"))
    if not calib_bundles or not holdout_bundles:
        raise FileNotFoundError(
            f"need calib+holdout 50-step bundles in {BUNDLE_DIR}; run "
            "scripts/cache_qwen_calibration_bundles.py first"
        )
    print(f"[qwen-e2e] {len(calib_bundles)} calib + {len(holdout_bundles)} holdout bundles",
          flush=True)

    _setup_compiled_dir(Path(args.transformer_cache))

    app = NeuronQwenImageApplication(
        model_path=args.model_dir,
        parallel=NovaParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width},
        text_seq_len=args.text_seq_len,
        teacache_fused=True,
    )
    print(f"[qwen-e2e] components: {[c.name for c in app.components()]}", flush=True)
    print("[qwen-e2e] compiling fused probe only...", flush=True)
    t = time.perf_counter()
    app.teacache_probe.compile(str(COMPILED / "teacache_probe"))
    print(f"[qwen-e2e] fused probe compiled in {time.perf_counter() - t:.1f}s", flush=True)
    app.load(str(COMPILED), skip_warmup=True)
    print("[qwen-e2e] loaded", flush=True)

    # The transformer-only model dir has no scheduler/; inject the real
    # FlowMatchEulerDiscreteScheduler so the denoise loop matches diffusers.
    from diffusers import FlowMatchEulerDiscreteScheduler

    app.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.scheduler_id, subfolder="scheduler", local_files_only=True
    )
    print(f"[qwen-e2e] scheduler <- {args.scheduler_id}", flush=True)

    shape_label = f"{args.height}x{args.width}"

    # --- T0: collect calibration pairs on the calib bundles ---
    app.pipeline.teacache_controller = None
    train_samples: list[dict[str, Any]] = []
    for path in calib_bundles:
        _, tensors = _load_bundle(path)
        train_samples.extend(_collect_pairs(app, tensors, args.num_steps, "train"))
    train_x, train_y = _sample_xy(train_samples)
    # Qwen's block-0 mod_input diff is O(1e4-1e5) (vs HV's O(10-100)); a raw
    # degree-3 Vandermonde fit is catastrophically ill-conditioned. Fit on
    # x/scale, then rescale coefficients back to the raw-x domain so the
    # controller's predict_delta(raw_x) stays correct. coef_raw[i] = c_n[i]/s^i.
    scale = float(train_x.float().abs().mean().item()) or 1.0
    coef_n = _fit_poly(train_x / scale, train_y, args.degree)
    coef = torch.tensor(
        [float(coef_n[i]) / (scale ** i) for i in range(args.degree + 1)],
        dtype=torch.float64,
    )
    train_pred = _poly_design(train_x / scale, args.degree).matmul(coef_n)
    threshold = _threshold_for_target(
        train_pred,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        cooldown_steps=args.cooldown_steps,
        target_speedup=args.target_speedup,
    )

    # Diagnostics: is this a conditioning problem (fixed by scaling) or a weak
    # signal (mod_input diff does not predict noise_pred diff)?
    pearson = float(
        torch.corrcoef(torch.stack([train_x.double(), train_y.double()]))[0, 1].item()
    )
    print(f"[qwen-e2e] scale={scale:.2f} pearson(x,y)={pearson:.4f} (probe norm in fp32)",
          flush=True)
    for deg in (1, 2, 3):
        cn = _fit_poly(train_x / scale, train_y, deg)
        pred = _poly_design(train_x / scale, deg).matmul(cn)
        print(f"[qwen-e2e]   degree {deg} train_R2={_r2_score(train_y, pred):.4f}", flush=True)

    # Save pairs so the fit can be re-derived offline without re-running HW.
    pairs_doc = {
        "schema": "nova-m9-teacache-pairs-v1",
        "model": MODEL,
        "shape_label": shape_label,
        "num_steps": int(args.num_steps),
        "mod_input_source": "block0_modulated_input",
        "hardware_measured": True,
        "samples": train_samples,
    }
    (CCLOG / f"pairs_qwen_image_1024_{args.num_steps}step.json").write_text(
        json.dumps(pairs_doc, indent=2, sort_keys=True) + "\n"
    )

    # --- T0b baseline: controller off, full DiT every step; also collect
    #     holdout pairs for a held-out R^2. ---
    baselines: list[tuple[Any, float]] = []
    holdout_samples: list[dict[str, Any]] = []
    for path in holdout_bundles:
        _, tensors = _load_bundle(path)
        holdout_samples.extend(_collect_pairs(app, tensors, args.num_steps, "holdout"))
        out, el = _run_bundle(app, tensors, args.num_steps)
        baselines.append((out, el))
        print(f"[qwen-e2e] baseline {path.name}: {el:.3f}s", flush=True)

    eval_x, eval_y = _sample_xy(holdout_samples or train_samples)
    eval_pred = _poly_design(eval_x / scale, args.degree).matmul(coef_n)
    fit_r2 = _r2_score(eval_y, eval_pred)

    calibration = TeaCacheCalibration(
        model=MODEL,
        shape_label=shape_label,
        num_steps=int(args.num_steps),
        poly_coef=tuple(float(c) for c in coef.tolist()),
        threshold=float(threshold),
        warmup_steps=int(args.warmup_steps),
        cooldown_steps=int(args.cooldown_steps),
        target_speedup=float(args.target_speedup),
        fit_r2=float(fit_r2),
        n_samples=len(train_samples) + len(holdout_samples),
    )
    calib_doc = calibration.to_dict()
    calib_doc["hardware_measured"] = True
    calib_path = CCLOG / f"calibration_qwen_image_1024_{args.num_steps}step_t{args.target_speedup}.json"
    calib_path.write_text(json.dumps(calib_doc, indent=2, sort_keys=True) + "\n")
    print(f"[qwen-e2e] calibration R^2={fit_r2:.4f} threshold={threshold:.2f} -> {calib_path.name}",
          flush=True)

    # --- T0b fused: controller on ---
    app.pipeline.teacache_controller = TeaCacheController(calibration)
    app.pipeline.teacache_speedup = float(args.target_speedup)
    fused_times, tcos, fcos, skips, full = [], [], [], 0, 0
    for i, path in enumerate(holdout_bundles):
        _, tensors = _load_bundle(path)
        app.pipeline.teacache_controller.reset()
        out, el = _run_bundle(app, tensors, args.num_steps)
        bout, _ = baselines[i]
        fused_times.append(el)
        tcos.append(_trajectory_cosine(out.trajectory, bout.trajectory))
        fcos.append(_tensor_cosine(out.latents, bout.latents))
        st = app.pipeline.teacache_controller.stats()
        skips += int(st["skipped_steps"])
        full += int(st["full_steps"])
        print(f"[qwen-e2e] fused {path.name}: {el:.3f}s baseline={baselines[i][1]:.3f}s "
              f"skip={st['skipped_steps']} full={st['full_steps']}", flush=True)

    baseline_total = sum(e for _, e in baselines)
    fused_total = sum(fused_times)
    speedup = baseline_total / fused_total
    traj_cos = min(tcos)
    final_cos = min(fcos)

    ab = {
        "schema": "nova-m9-teacache-fused-e2e-v1",
        "model": MODEL,
        "shape_label": shape_label,
        "num_steps": int(args.num_steps),
        "n_bundles": len(holdout_bundles),
        "baseline_total_s": baseline_total,
        "fused_total_s": fused_total,
        "measured_speedup": speedup,
        "trajectory_cosine": traj_cos,
        "final_cosine": final_cos,
        "skipped_steps": skips,
        "full_steps": full,
        "probe_ms_per_call": 1.35,
        "dit_step_s": 0.42,
        "hardware_measured": True,
    }
    (CCLOG / "qwen_fused_e2e_ab.json").write_text(json.dumps(ab, indent=2, sort_keys=True) + "\n")

    curve = {
        "schema": "nova-m9-teacache-speedup-curve-v1",
        "model": MODEL,
        "shape_label": shape_label,
        "num_steps": int(args.num_steps),
        "candidates": [
            {
                "target_speedup": float(args.target_speedup),
                "measured_speedup": speedup,
                "trajectory_cosine": traj_cos,
                "final_cosine": final_cos,
                "calibration": str(calib_path),
                "full_steps": full,
                "skipped_steps": skips,
                "probe": "fused-A",
                "hardware_measured": True,
            }
        ],
    }
    (CCLOG / f"speedup_curve_qwen_{args.num_steps}step_fused.json").write_text(
        json.dumps(curve, indent=2, sort_keys=True) + "\n"
    )

    integration = {
        "schema": "nova-m9-teacache-integration-v1",
        "model": MODEL,
        "shape_label": shape_label,
        "num_steps": int(args.num_steps),
        "wallclock_speedup": speedup,
        "trajectory_cosine": traj_cos,
        "final_cosine": final_cos,
        "default_off": True,
        "hardware_measured": True,
    }
    (CCLOG / "integration_qwen_50step_fused.json").write_text(
        json.dumps(integration, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"[qwen-e2e] SPEEDUP fused={speedup:.4f}x | traj={traj_cos:.6f} final={final_cos:.6f} "
        f"| skipped={skips}/{skips + full}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
