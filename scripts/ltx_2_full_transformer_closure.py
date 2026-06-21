#!/usr/bin/env python3
"""Run the LTX-2 full-snapshot transformer closure path.

This script closes the transformer boundary when a local LTX-2 snapshot is
available:

1. verify the local snapshot has the host-side text/connectors/scheduler and
   transformer files,
2. cache real LTX-2 prompt embeddings, packed video/audio latents, coords, and
   timesteps,
3. run Trainium-vs-CPU transformer parity on the cached inputs.

Video VAE, audio VAE, and vocoder decode remain a separate end-to-end closure
gate unless ``--require-decode-components`` is supplied for preflight.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime_python() -> None:
    if Path(sys.executable) == NEURON_PYTHON or not NEURON_PYTHON.exists():
        return
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        env = os.environ.copy()
        env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("DIFFLET_LTX_2_MODEL_DIR", ""),
        help="Local LTX-2 snapshot dir. Env: DIFFLET_LTX_2_MODEL_DIR.",
    )
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--text-seq-len", type=int, default=1024)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bundle",
        default=".difflet-cache/ltx_2_dit_inputs/full_512x768x121_4step.safetensors",
    )
    parser.add_argument("--cache-dir", default=".difflet-cache/ltx_2_transformer_full")
    parser.add_argument(
        "--metrics-out",
        default="/tmp/difflet_ltx_2_full_transformer_parity_metrics.json",
    )
    parser.add_argument("--reference-mode", choices=("trace", "diffusers"), default="trace")
    parser.add_argument("--min-video-cosine", type=float, default=0.999)
    parser.add_argument("--min-audio-cosine", type=float, default=0.999)
    parser.add_argument("--max-video-mean-abs", type=float, default=None)
    parser.add_argument("--max-audio-mean-abs", type=float, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-decode-components", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=0)
    return parser


def _has_any(parent: Path, names: tuple[str, ...]) -> bool:
    return any((parent / name).exists() for name in names)


def _has_weights(parent: Path) -> bool:
    return _has_any(
        parent,
        (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.safetensors.index.json",
            "model.safetensors",
            "model.safetensors.index.json",
        ),
    )


def _snapshot_preflight(
    model_dir: Path,
    *,
    require_decode_components: bool,
) -> tuple[bool, list[str]]:
    required = [
        ("snapshot dir", model_dir.exists()),
        ("transformer/config.json", (model_dir / "transformer" / "config.json").exists()),
        ("transformer weights", _has_weights(model_dir / "transformer")),
        (
            "scheduler/scheduler_config.json",
            (model_dir / "scheduler" / "scheduler_config.json").exists(),
        ),
        ("text_encoder/config.json", (model_dir / "text_encoder" / "config.json").exists()),
        ("text_encoder weights", _has_weights(model_dir / "text_encoder")),
        (
            "tokenizer files",
            _has_any(
                model_dir / "tokenizer",
                ("tokenizer.json", "tokenizer_config.json", "spiece.model"),
            ),
        ),
        ("connectors/config.json", (model_dir / "connectors" / "config.json").exists()),
        ("connectors weights", _has_weights(model_dir / "connectors")),
    ]
    optional_decode = [
        ("vae/config.json", (model_dir / "vae" / "config.json").exists()),
        ("vae weights", _has_weights(model_dir / "vae")),
        ("audio_vae/config.json", (model_dir / "audio_vae" / "config.json").exists()),
        ("audio_vae weights", _has_weights(model_dir / "audio_vae")),
        ("vocoder/config.json", (model_dir / "vocoder" / "config.json").exists()),
        ("vocoder weights", _has_weights(model_dir / "vocoder")),
    ]
    checks = required + optional_decode
    lines = [f"{'OK' if passed else 'MISSING'} {name}" for name, passed in checks]
    required_ok = all(passed for _name, passed in required)
    decode_ok = all(passed for _name, passed in optional_decode)
    return required_ok and (decode_ok or not require_decode_components), lines


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("[ltx2-full] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _default_prompt() -> list[str]:
    return ["a close-up cinematic shot of a glass teapot on a wooden table"]


def main() -> int:
    ensure_runtime_python()
    args = build_parser().parse_args()
    if not args.model_dir:
        print(
            "[ltx2-full] --model-dir or DIFFLET_LTX_2_MODEL_DIR is required",
            file=sys.stderr,
        )
        return 2

    model_dir = Path(args.model_dir).expanduser().resolve()
    ok, lines = _snapshot_preflight(
        model_dir,
        require_decode_components=args.require_decode_components,
    )
    print(f"[ltx2-full] model_dir = {model_dir}")
    for line in lines:
        print(f"[ltx2-full] preflight {line}")
    if not ok:
        print("[ltx2-full] preflight FAIL: local LTX-2 snapshot is incomplete", file=sys.stderr)
        return 2
    if args.preflight_only:
        print("[ltx2-full] preflight PASS")
        return 0

    env = os.environ.copy()
    env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("DIFFLET_BACKEND", "trainium")
    env.setdefault("NEURON_RT_NUM_CORES", str(args.tp_degree))
    env.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")

    bundle = Path(args.bundle)
    prompt = args.prompt or _default_prompt()
    if args.force_cache or not bundle.exists():
        cache_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "ltx_2_cache_dit_inputs.py"),
            "--model-id",
            str(model_dir),
            "--local-files-only",
            "--output",
            str(bundle),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num-frames",
            str(args.num_frames),
            "--frame-rate",
            str(args.frame_rate),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--text-seq-len",
            str(args.text_seq_len),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        for item in prompt:
            cache_cmd.extend(["--prompt", item])
        _run(cache_cmd, env=env)
    else:
        print(f"[ltx2-full] reuse bundle = {bundle}")

    parity_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ltx_2_transformer_parity.py"),
        "--model-dir",
        str(model_dir),
        "--bundle",
        str(bundle),
        "--cache-dir",
        args.cache_dir,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-frames",
        str(args.num_frames),
        "--tp-degree",
        str(args.tp_degree),
        "--reference-mode",
        args.reference_mode,
        "--metrics-out",
        args.metrics_out,
        "--min-video-cosine",
        str(args.min_video_cosine),
        "--min-audio-cosine",
        str(args.min_audio_cosine),
    ]
    if args.max_video_mean_abs is not None:
        parity_cmd.extend(["--max-video-mean-abs", str(args.max_video_mean_abs)])
    if args.max_audio_mean_abs is not None:
        parity_cmd.extend(["--max-audio-mean-abs", str(args.max_audio_mean_abs)])
    if args.force_compile:
        parity_cmd.append("--force-compile")
    if args.skip_compile:
        parity_cmd.append("--skip-compile")
    if args.skip_warmup:
        parity_cmd.append("--skip-warmup")
    if args.num_threads > 0:
        parity_cmd.extend(["--num-threads", str(args.num_threads)])
    _run(parity_cmd, env=env)

    metrics_path = Path(args.metrics_out)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(
            "[ltx2-full] parity PASS "
            f"video_cosine={metrics['video_cosine']:.10f} "
            f"audio_cosine={metrics['audio_cosine']:.10f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
