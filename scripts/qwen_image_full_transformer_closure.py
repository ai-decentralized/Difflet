#!/usr/bin/env python3
"""Run the Qwen-Image full-snapshot transformer closure path.

This script intentionally closes only the transformer boundary:

1. verify a local Qwen-Image snapshot has the text/scheduler/transformer files,
2. cache real Qwen2.5-VL prompt embeddings and packed latents,
3. run Trainium-vs-CPU transformer parity on the cached inputs.

Qwen VAE decode remains a separate closure item.
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
        default=os.environ.get("NOVA_QWEN_IMAGE_MODEL_DIR", ""),
        help="Local Qwen-Image snapshot dir. Env: NOVA_QWEN_IMAGE_MODEL_DIR.",
    )
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--text-seq-len", type=int, default=1024)
    parser.add_argument(
        "--pad-to-text-seq-len",
        action="store_true",
        help="Pad cached text embeddings to --text-seq-len instead of compiling active length.",
    )
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument(
        "--bundle",
        default=".nova-cache/qwen_image_dit_inputs/full_1024_4step.safetensors",
    )
    parser.add_argument(
        "--cache-dir",
        default=".nova-cache/qwen_image_transformer_full",
    )
    parser.add_argument(
        "--metrics-out",
        default="/tmp/nova_qwen_image_full_transformer_parity_metrics.json",
    )
    parser.add_argument("--reference-mode", choices=("trace", "diffusers"), default="trace")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-mean-abs", type=float, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=0)
    return parser


def _has_any(parent: Path, names: tuple[str, ...]) -> bool:
    return any((parent / name).exists() for name in names)


def _snapshot_preflight(model_dir: Path) -> tuple[bool, list[str]]:
    checks = [
        ("snapshot dir", model_dir.exists()),
        ("transformer/config.json", (model_dir / "transformer" / "config.json").exists()),
        (
            "transformer weights",
            _has_any(
                model_dir / "transformer",
                (
                    "diffusion_pytorch_model.safetensors",
                    "diffusion_pytorch_model.safetensors.index.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                ),
            ),
        ),
        ("scheduler/scheduler_config.json", (model_dir / "scheduler" / "scheduler_config.json").exists()),
        ("text_encoder/config.json", (model_dir / "text_encoder" / "config.json").exists()),
        (
            "text_encoder weights",
            _has_any(
                model_dir / "text_encoder",
                (
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "diffusion_pytorch_model.safetensors",
                    "diffusion_pytorch_model.safetensors.index.json",
                ),
            ),
        ),
        (
            "tokenizer files",
            _has_any(
                model_dir / "tokenizer",
                ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
            ),
        ),
    ]
    lines = [f"{'OK' if passed else 'MISSING'} {name}" for name, passed in checks]
    return all(passed for _name, passed in checks), lines


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("[qwen-full] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _default_prompt() -> list[str]:
    return ["a small red cabin beside a lake, crisp morning light"]


def main() -> int:
    ensure_runtime_python()
    args = build_parser().parse_args()
    if not args.model_dir:
        print(
            "[qwen-full] --model-dir or NOVA_QWEN_IMAGE_MODEL_DIR is required for full closure",
            file=sys.stderr,
        )
        return 2

    model_dir = Path(args.model_dir).expanduser().resolve()
    ok, lines = _snapshot_preflight(model_dir)
    print(f"[qwen-full] model_dir = {model_dir}")
    for line in lines:
        print(f"[qwen-full] preflight {line}")
    if not ok:
        print("[qwen-full] preflight FAIL: full local Qwen-Image snapshot is incomplete", file=sys.stderr)
        return 2
    if args.preflight_only:
        print("[qwen-full] preflight PASS")
        return 0

    env = os.environ.copy()
    env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("NOVA_BACKEND", "trainium")
    env.setdefault("NEURON_RT_NUM_CORES", str(args.tp_degree))
    env.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")

    bundle = Path(args.bundle)
    prompt = args.prompt or _default_prompt()
    if args.force_cache or not bundle.exists():
        cache_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "qwen_image_cache_dit_inputs.py"),
            "--model-id",
            str(model_dir),
            "--local-files-only",
            "--output",
            str(bundle),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--text-seq-len",
            str(args.text_seq_len),
            "--seed",
            str(args.seed),
            "--guidance-scale",
            str(args.guidance_scale),
            "--device",
            args.device,
        ]
        if args.pad_to_text_seq_len:
            cache_cmd.append("--pad-to-text-seq-len")
        for item in prompt:
            cache_cmd.extend(["--prompt", item])
        _run(cache_cmd, env=env)
    else:
        print(f"[qwen-full] reuse bundle = {bundle}")

    parity_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "qwen_image_transformer_parity.py"),
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
        "--tp-degree",
        str(args.tp_degree),
        "--reference-mode",
        args.reference_mode,
        "--metrics-out",
        args.metrics_out,
        "--min-cosine",
        str(args.min_cosine),
    ]
    if args.max_mean_abs is not None:
        parity_cmd.extend(["--max-mean-abs", str(args.max_mean_abs)])
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
            "[qwen-full] parity PASS "
            f"cosine={metrics['cosine']:.10f} mean_abs={metrics['mean_abs']:.6e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
