#!/usr/bin/env python3
"""cclog 83 / Path A: Qwen-Image TeaCache with the relative-L1 signal + the
vLLM-Omni rescaling polynomial + the original-TeaCache accumulator.

cclog 82 falsified TeaCache on Qwen using an *absolute L2* block-0 signal
(Pearson 0.30). vLLM-Omni reports 1.91x on the same model — proof the recipe
works, with the *relative-L1* signal. This run:

  1. confirms the signal: Pearson(rel_l1_mod, rel_l1_noise) on calib bundles;
  2. sweeps the accumulator threshold (vLLM poly, fixed) on the holdout
     bundles, baseline vs fused, reporting speedup + trajectory/final cosine.

The probe now returns relative L1 (mean|mod-prev|/mean|prev|) — see
difflet/backends/trainium/qwen_image/teacache_probe_fused.py.
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

from difflet.models.qwen_image.application import QwenImageDiTInputBundle  # noqa: E402
from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController
from scripts.run_qwen_teacache_e2e import (
    BUNDLE_DIR,
    CCLOG,
    COMPILED,
    DEFAULT_MODEL_DIR,
    DEFAULT_TRANSFORMER_CACHE,
    MODEL,
    _build_bundle,
    _load_bundle,
    _run_bundle,
    _setup_compiled_dir,
    _setup_scheduler,
    _tensor_cosine,
    _trajectory_cosine,
)
from scripts.calibrate_teacache import _mark_step

# vLLM-Omni QwenImageTransformer2DModel coefficients are np.poly1d order
# (highest degree first): -450 x^4 + 280 x^3 - 45 x^2 + 3.2 x - 0.02.
# predict_delta() evaluates ascending powers, so store constant-first.
VLLM_POLY_ASCENDING = (-0.02, 3.2, -45.0, 280.0, -450.0)


def _pearson(a: list[float], b: list[float]) -> float:
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    return float(torch.corrcoef(torch.stack([ta, tb]))[0, 1].item())


def _collect_rel_corr(app, tensors, num_steps: int) -> list[tuple[float, float]]:
    """Per step, the device relative-L1 of the block-0 mod_input and the host
    relative-L1 of the noise_pred. Confirms the signal predicts the output."""
    from difflet.models.qwen_image.pipeline import _batch_timestep, _component_dtype, _first_tensor

    pipe = app.pipeline
    bundle = _build_bundle(tensors)
    image_seq_len = int(tensors["latents_init"].shape[1])
    timesteps = _setup_scheduler(pipe, num_steps, image_seq_len)
    latents = bundle.hidden_states
    prev_np: torch.Tensor | None = None
    pairs: list[tuple[float, float]] = []
    for timestep in timesteps:
        model_dtype = _component_dtype(pipe.transformer, pipe.dtype)
        tb = _batch_timestep(timestep, latents.shape[0], latents.device, model_dtype)
        mb = QwenImageDiTInputBundle(
            hidden_states=latents.to(dtype=model_dtype),
            timestep=(tb / 1000.0).to(dtype=model_dtype),  # diffusers feeds timestep/1000
            encoder_hidden_states=bundle.encoder_hidden_states.to(dtype=model_dtype),
            encoder_hidden_states_mask=bundle.encoder_hidden_states_mask,
            guidance=bundle.guidance.to(dtype=model_dtype),
        )
        rel_l1_mod = float(app.teacache_delta(mb).detach().cpu().item())
        np_pred = _first_tensor(app(mb)).detach().float().cpu()
        _mark_step()
        if prev_np is not None:
            rel_l1_np = float(
                (np_pred - prev_np).abs().mean().item() / (prev_np.abs().mean().item() + 1e-8)
            )
            pairs.append((rel_l1_mod, rel_l1_np))
        prev_np = np_pred
        latents = pipe._scheduler_step(np_pred, timestep, latents, int(timesteps.numel()))
        _mark_step()
    return pairs


def _build_and_load_app(args):
    from difflet.models.qwen_image.application import NeuronQwenImageApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    _setup_compiled_dir(Path(args.transformer_cache))
    app = NeuronQwenImageApplication(
        model_path=args.model_dir,
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width},
        text_seq_len=args.text_seq_len,
        teacache_fused=True,
    )
    print(f"[relL1] components: {[c.name for c in app.components()]}", flush=True)
    print("[relL1] compiling fused probe (relative-L1)...", flush=True)
    t = time.perf_counter()
    app.teacache_probe.compile(str(COMPILED / "teacache_probe"))
    print(f"[relL1] probe compiled in {time.perf_counter() - t:.1f}s", flush=True)
    app.load(str(COMPILED), skip_warmup=True)
    from diffusers import FlowMatchEulerDiscreteScheduler

    app.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.scheduler_id, subfolder="scheduler", local_files_only=True
    )
    print("[relL1] loaded + scheduler injected", flush=True)
    return app


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--scheduler-id", default="Qwen/Qwen-Image")
    p.add_argument("--transformer-cache", default=str(DEFAULT_TRANSFORMER_CACHE))
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--text-seq-len", type=int, default=1024)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--cooldown-steps", type=int, default=2)
    p.add_argument("--n-calib", type=int, default=4, help="calib bundles for the correlation check")
    p.add_argument("--thresholds", type=float, nargs="*", default=[0.1, 0.15, 0.2, 0.3])
    return p.parse_args()


def _calibration(threshold: float, num_steps: int, warmup: int, cooldown: int) -> TeaCacheCalibration:
    return TeaCacheCalibration(
        model=MODEL,
        shape_label="1024x1024",
        num_steps=int(num_steps),
        poly_coef=VLLM_POLY_ASCENDING,
        threshold=float(threshold),
        warmup_steps=int(warmup),
        cooldown_steps=int(cooldown),
        accumulate=True,
        mod_input_source="block0_modulated_input",
    )


def main() -> int:
    args = _parse_args()
    calib_bundles = sorted(BUNDLE_DIR.glob("calibration_*.safetensors"))[: args.n_calib]
    holdout_bundles = sorted(BUNDLE_DIR.glob("holdout_*.safetensors"))
    if not calib_bundles or not holdout_bundles:
        raise FileNotFoundError(f"need bundles in {BUNDLE_DIR}")

    app = _build_and_load_app(args)

    # --- signal confirmation: rel_l1_mod vs rel_l1_noise ---
    app.pipeline.teacache_controller = None
    xs: list[float] = []
    ys: list[float] = []
    for path in calib_bundles:
        _, tensors = _load_bundle(path)
        for xm, yn in _collect_rel_corr(app, tensors, args.num_steps):
            xs.append(xm)
            ys.append(yn)
    print(
        f"[relL1] SIGNAL n={len(xs)} Pearson(rel_l1_mod, rel_l1_noise)={_pearson(xs, ys):.4f} "
        f"| rel_l1_mod mean={sum(xs)/len(xs):.4f} range[{min(xs):.4f},{max(xs):.4f}]",
        flush=True,
    )

    # --- baseline holdout ---
    baselines: list[tuple[Any, float]] = []
    for path in holdout_bundles:
        _, tensors = _load_bundle(path)
        out, el = _run_bundle(app, tensors, args.num_steps)
        baselines.append((out, el))
        print(f"[relL1] baseline {path.name}: {el:.3f}s", flush=True)
    base_total = sum(e for _, e in baselines)

    # --- threshold sweep (vLLM poly + accumulator) ---
    candidates = []
    for th in args.thresholds:
        calib = _calibration(th, args.num_steps, args.warmup_steps, args.cooldown_steps)
        app.pipeline.teacache_controller = TeaCacheController(calib)
        app.pipeline.teacache_speedup = None
        times, tcos, fcos, skips, full = [], [], [], 0, 0
        for i, path in enumerate(holdout_bundles):
            _, tensors = _load_bundle(path)
            app.pipeline.teacache_controller.reset()
            out, el = _run_bundle(app, tensors, args.num_steps)
            bout, _ = baselines[i]
            times.append(el)
            tcos.append(_trajectory_cosine(out.trajectory, bout.trajectory))
            fcos.append(_tensor_cosine(out.latents, bout.latents))
            st = app.pipeline.teacache_controller.stats()
            skips += int(st["skipped_steps"])
            full += int(st["full_steps"])
        fused_total = sum(times)
        cand = {
            "threshold": float(th),
            "measured_speedup": base_total / fused_total,
            "trajectory_cosine": min(tcos),
            "final_cosine": min(fcos),
            "skipped_steps": skips,
            "full_steps": full,
            "hardware_measured": True,
        }
        candidates.append(cand)
        print(
            f"[relL1] th={th:.3f} speedup={cand['measured_speedup']:.3f}x "
            f"traj={cand['trajectory_cosine']:.4f} final={cand['final_cosine']:.4f} "
            f"skip={skips}/{skips + full}",
            flush=True,
        )

    doc = {
        "schema": "difflet-m9-teacache-relL1-sweep-v1",
        "model": MODEL,
        "shape_label": "1024x1024",
        "num_steps": int(args.num_steps),
        "signal": "relative_l1_block0_modulated_input",
        "poly": "vllm_omni_QwenImageTransformer2DModel",
        "signal_pearson": _pearson(xs, ys),
        "baseline_total_s": base_total,
        "candidates": candidates,
        "hardware_measured": True,
    }
    (CCLOG / "qwen_relL1_sweep.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    best = max(
        (c for c in candidates if c["trajectory_cosine"] >= 0.99),
        key=lambda c: c["measured_speedup"],
        default=None,
    )
    print(f"[relL1] best @ cosine>=0.99: {best}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
