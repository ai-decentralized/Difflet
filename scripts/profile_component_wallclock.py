#!/usr/bin/env python3
"""F2.0 per-component wall-clock profiler for DiT/VAE separation.

Measures the end-to-end wall-clock share of each pipeline component (text
encoder / DiT denoise loop / VAE decode) so cclog 65 can apply the
"VAE share >= 20%" gate before any F2.1 implementation work starts.

The text encoder is assumed to be prebaked into the cached DiT input bundle
(``encoder_hidden_states`` / ``pooled_projections`` / etc. arrive already
encoded), so text-encoder wall-clock for this measurement is 0 by construction.
This is acceptable for F2.0 because F2 targets the DiT/VAE asymmetry, not the
text encoder.

For HunyuanVideo the script drives the pipeline's own ``_denoise`` and
``_decode_latents`` boundaries with ``xm.mark_step()`` fences between them so
device-side work has actually committed before the boundary timer stops.

Example:
    NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \\
    python scripts/profile_component_wallclock.py \\
        --model hunyuan-video \\
        --source-dir /home/ubuntu/.cache/huggingface/hub/hunyuanvideo-real \\
        --compiled-dir .nova-cache/f3_hunyuan_n4_4d8s1r/compiled \\
        --bundle .nova-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors \\
        --tp-degree 4 --skip-warmup \\
        --metrics-out cclogs/m8-dit-vae-separation/profile_hv_n4_4d8s1r_components.json
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
DEFAULT_OUT_DIR = ROOT / "cclogs" / "m8-dit-vae-separation"

SCHEMA = "nova-f2-0-component-wallclock-v1"
GATE_THRESHOLD = 0.20


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
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402


def _mark_step() -> float:
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return 0.0
    start = time.perf_counter()
    xm.mark_step()
    return time.perf_counter() - start


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        choices=("hunyuan-video", "hunyuan-video15"),
        help="Which Nova application to drive. F2 is video-only (D1).",
    )
    p.add_argument("--source-dir", required=True, help="HF source path (transformer + vae)")
    p.add_argument("--compiled-dir", required=True, help="Nova compiled cache parent dir")
    p.add_argument("--bundle", required=True, help="Cached DiT input safetensors")
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument(
        "--enable-trainium-vae",
        action="store_true",
        help="Use compiled Trainium VAE (must exist under --compiled-dir). Default: HF CPU VAE.",
    )
    p.add_argument("--metrics-out", required=True)
    return p.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_hunyuan_app(args: argparse.Namespace, meta: dict[str, Any]):
    from nova.models.hunyuan_video.application import NeuronHunyuanVideoApplication
    from nova.pipeline.parallel_config import NovaParallelConfig

    return NeuronHunyuanVideoApplication(
        model_path=args.source_dir,
        parallel=NovaParallelConfig(tp_degree=args.tp_degree, cp_enabled=False),
        dtype=_dtype_from_name(args.dtype),
        shape={
            "height": int(meta["height"]),
            "width": int(meta["width"]),
            "num_frames": int(meta["num_frames"]),
        },
        text_seq_len=int(meta["text_seq_len"]),
        enable_vae_decoder=bool(args.enable_trainium_vae),
    )


def _drive_hunyuan(
    args: argparse.Namespace, meta: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> dict[str, Any]:
    from nova.models.hunyuan_video.application import HunyuanVideoDiTInputBundle

    app = _load_hunyuan_app(args, meta)
    t_load_start = time.perf_counter()
    app.load(args.compiled_dir, skip_warmup=bool(args.skip_warmup))
    load_elapsed = time.perf_counter() - t_load_start
    print(f"[f2.0] load elapsed = {load_elapsed:.3f}s", flush=True)

    pipe = app.pipeline
    if pipe.transformer is None:
        raise RuntimeError("HV pipeline transformer is None — compiled artifact incomplete")

    num_steps = int(args.num_inference_steps or meta["num_inference_steps"])
    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=tensors["latents_init"],
        timestep=tensors["timesteps"][:1].clone(),
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_attention_mask=tensors["encoder_attention_mask"],
        pooled_projections=tensors["pooled_projections"],
        guidance=tensors["guidance"],
    )

    e2e_start = time.perf_counter()

    # ----- DiT denoise loop -----
    dit_phase_start = time.perf_counter()
    timesteps = pipe._prepare_explicit_timesteps(  # noqa: SLF001
        tensors["timesteps"], device=tensors["latents_init"].device
    )
    per_step_s: list[float] = []
    latents = bundle.hidden_states
    from nova.models.hunyuan_video.pipeline import (  # noqa: E402
        _batch_timestep,
        _component_dtype,
        _first_tensor,
    )

    for step_idx, timestep in enumerate(timesteps):
        step_start = time.perf_counter()
        model_dtype = _component_dtype(pipe.transformer, pipe.dtype)
        timestep_batch = _batch_timestep(
            timestep,
            batch_size=latents.shape[0],
            device=latents.device,
            dtype=model_dtype,
        )
        model_bundle = HunyuanVideoDiTInputBundle(
            hidden_states=latents.to(dtype=model_dtype),
            timestep=timestep_batch,
            encoder_hidden_states=bundle.encoder_hidden_states.to(dtype=model_dtype),
            encoder_attention_mask=bundle.encoder_attention_mask,
            pooled_projections=bundle.pooled_projections.to(dtype=model_dtype),
            guidance=bundle.guidance.to(dtype=model_dtype),
        )
        noise_pred = _first_tensor(pipe.transformer(model_bundle))
        latents = pipe._scheduler_step(  # noqa: SLF001
            noise_pred, timestep, latents, int(timesteps.shape[0])
        )
        _mark_step()
        per_step_s.append(time.perf_counter() - step_start)
        print(
            f"[f2.0] dit step {step_idx + 1}/{len(timesteps)} = {per_step_s[-1]:.4f}s",
            flush=True,
        )

    dit_phase_elapsed = time.perf_counter() - dit_phase_start

    # ----- VAE decode -----
    vae_start = time.perf_counter()
    frames = pipe._decode_latents(latents)  # noqa: SLF001
    _mark_step()
    if torch.is_tensor(frames):
        # force materialization in case the VAE returned an XLA tensor
        _ = frames.shape
    vae_elapsed = time.perf_counter() - vae_start

    e2e_elapsed = time.perf_counter() - e2e_start

    print(
        f"[f2.0] dit_total = {dit_phase_elapsed:.4f}s | "
        f"vae_total = {vae_elapsed:.4f}s | e2e = {e2e_elapsed:.4f}s",
        flush=True,
    )

    return {
        "load_compile_elapsed_s": load_elapsed,
        "num_steps": int(num_steps),
        "shape": {
            "height": int(meta["height"]),
            "width": int(meta["width"]),
            "num_frames": int(meta["num_frames"]),
        },
        "text_seq_len": int(meta["text_seq_len"]),
        "tp_degree": int(args.tp_degree),
        "dtype": args.dtype,
        "enable_trainium_vae": bool(args.enable_trainium_vae),
        "frames_shape": list(frames.shape) if torch.is_tensor(frames) else None,
        "frames_dtype": str(frames.dtype) if torch.is_tensor(frames) else None,
        "dit_total_s": float(dit_phase_elapsed),
        "dit_per_step_s": [float(x) for x in per_step_s],
        "dit_mean_step_s": float(sum(per_step_s) / max(len(per_step_s), 1)),
        "vae_total_s": float(vae_elapsed),
        "text_encoder_total_s": 0.0,
        "text_encoder_source": "prebaked_in_bundle",
        "e2e_total_s": float(e2e_elapsed),
        "vae_path": "trainium" if args.enable_trainium_vae else "host_cpu_hf",
    }


def _shares(metrics: dict[str, Any]) -> dict[str, Any]:
    e2e = float(metrics["e2e_total_s"])
    vae = float(metrics["vae_total_s"])
    dit = float(metrics["dit_total_s"])
    txt = float(metrics["text_encoder_total_s"])
    denom = max(e2e, 1e-12)
    return {
        "vae_share": vae / denom,
        "dit_share": dit / denom,
        "text_encoder_share": txt / denom,
        "unaccounted_share": max(1.0 - (vae + dit + txt) / denom, 0.0),
        "gate_threshold": GATE_THRESHOLD,
        "passes_threshold": (vae / denom) >= GATE_THRESHOLD,
    }


def main() -> int:
    args = _parse_args()
    bundle_path = Path(args.bundle)
    meta_path = Path(args.bundle + ".meta.json")
    if not meta_path.exists():
        print(f"[f2.0] missing meta sidecar: {meta_path}", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text())
    tensors = {
        k: v.to(dtype=_dtype_from_name(args.dtype)) if v.is_floating_point() else v
        for k, v in load_safetensors_file(str(bundle_path), device="cpu").items()
    }

    if args.model in ("hunyuan-video", "hunyuan-video15"):
        run_metrics = _drive_hunyuan(args, meta, tensors)
    else:
        raise SystemExit(f"unsupported --model {args.model!r}")

    shares = _shares(run_metrics)
    out = {
        "schema": SCHEMA,
        "model": args.model,
        "source_dir": str(args.source_dir),
        "compiled_dir": str(args.compiled_dir),
        "bundle": str(args.bundle),
        "command_args": vars(args),
        "components": run_metrics,
        "shares": shares,
    }

    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[f2.0] metrics -> {out_path}", flush=True)
    print(
        f"[f2.0] vae_share = {shares['vae_share']:.4%} | "
        f"dit_share = {shares['dit_share']:.4%} | "
        f"passes_threshold = {shares['passes_threshold']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
