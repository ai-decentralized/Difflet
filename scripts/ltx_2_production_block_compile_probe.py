#!/usr/bin/env python3
"""Probe LTX-2 production-shape compile capacity with fewer transformer layers.

The full 48-layer single-component transformer exceeds the Neuron compiler
instruction-count limit. This script copies a real LTX-2 transformer config,
overrides ``num_layers``, and compiles the resulting production-shape module
with synthetic weights. It is a capacity probe for deciding whether blockwise
segmentation is sufficient.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

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
        default=os.environ.get("NOVA_LTX_2_MODEL_DIR", ""),
        help="Local LTX-2 snapshot dir. Env: NOVA_LTX_2_MODEL_DIR.",
    )
    parser.add_argument("--work-dir", default="/tmp/nova_ltx2_production_block_probe_model")
    parser.add_argument("--cache-dir", default="/tmp/nova_ltx2_production_block_probe_cache")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--audio-num-frames", type=int, default=126)
    parser.add_argument("--text-seq-len", type=int, default=1024)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser


def _write_probe_model(args: argparse.Namespace) -> Path:
    if not args.model_dir:
        raise ValueError("--model-dir or NOVA_LTX_2_MODEL_DIR is required")
    source = Path(args.model_dir).expanduser().resolve() / "transformer" / "config.json"
    if not source.exists():
        raise FileNotFoundError(f"missing transformer config: {source}")

    work_dir = Path(args.work_dir).expanduser().resolve()
    if args.force_clean and work_dir.exists():
        shutil.rmtree(work_dir)
    transformer_dir = work_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    config["num_layers"] = int(args.num_layers)
    (transformer_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return work_dir


def main() -> int:
    ensure_runtime_python()
    args = build_parser().parse_args()

    import torch

    from nova import NovaParallelConfig, NovaPipeline

    work_dir = _write_probe_model(args)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    pipe = NovaPipeline.from_pretrained(
        str(work_dir),
        model_type="ltx_2",
        parallel=NovaParallelConfig(tp_degree=args.tp_degree),
        dtype=dtype,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        compile_cache_dir=args.cache_dir,
        force_compile=args.force_compile,
        load=not args.skip_load,
        skip_warmup=args.skip_warmup,
        application_kwargs={
            "audio_num_frames": args.audio_num_frames,
            "text_seq_len": args.text_seq_len,
        },
    )
    print(f"[ltx2-block-probe] model_dir={work_dir}")
    print(f"[ltx2-block-probe] compiled_path={pipe.compiled_path}")
    print(f"[ltx2-block-probe] layers={args.num_layers} tp={args.tp_degree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
