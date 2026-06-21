#!/usr/bin/env python3
"""Compile the LTX-2 segmented transformer block boundary at production shape."""

from __future__ import annotations

import argparse
import os
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
    parser.add_argument("--cache-dir", default="/tmp/difflet_ltx2_segmented_block_probe_cache")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--audio-num-frames", type=int, default=126)
    parser.add_argument("--text-seq-len", type=int, default=1024)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser


def main() -> int:
    ensure_runtime_python()
    args = build_parser().parse_args()
    if not args.model_dir:
        raise ValueError("--model-dir or DIFFLET_LTX_2_MODEL_DIR is required")

    import torch

    from difflet import DiffletParallelConfig, DiffletPipeline

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    pipe = DiffletPipeline.from_pretrained(
        str(Path(args.model_dir).expanduser().resolve()),
        model_type="ltx_2",
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=dtype,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        compile_cache_dir=args.cache_dir,
        force_compile=args.force_compile,
        local_files_only=True,
        load=not args.skip_load,
        skip_warmup=args.skip_warmup,
        application_kwargs={
            "transformer_mode": "segmented",
            "audio_num_frames": args.audio_num_frames,
            "text_seq_len": args.text_seq_len,
        },
    )
    print(f"[ltx2-segmented-block-probe] model_dir={pipe.model_path}")
    print(f"[ltx2-segmented-block-probe] compiled_path={pipe.compiled_path}")
    print(f"[ltx2-segmented-block-probe] tp={args.tp_degree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
