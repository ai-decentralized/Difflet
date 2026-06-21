#!/usr/bin/env python3
"""Stage 2 of cclog 72 implementation verification.

Compiles ONLY the HunyuanVideo TeaCache probe NEFF (block-0 modulated input
path + device-side L2 diff). Does NOT recompile the full DiT NEFF — the
existing artifact at ``.difflet-cache/f3_hunyuan_n4_4d8s1r/compiled/transformer/``
is preserved.

Mechanism: instantiate ``NeuronHunyuanVideoBackboneApplication``, then pop
the DiT wrapper from ``self.models`` so the compile path only traces the
probe. Output goes to a separate directory so the existing DiT artifact is
untouched.

Run TTY-foreground per cclog 70 §"Guard" and cclog 72 risk-register
mitigation. Watch ``dmesg -w`` in another shell.

Example:
    PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \\
    PYTHONPATH=. NEURON_RT_NUM_CORES=4 \\
    /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python \\
      scripts/compile_hv_teacache_probe_only.py \\
        --source-dir .difflet-cache/f3_hunyuan_n4_4d8s1r/source \\
        --output-dir .difflet-cache/f3_hunyuan_n4_4d8s1r/compiled_probe \\
        --tp-degree 4
"""

from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir",
        required=True,
        help="HF source directory (must contain transformer/config.json + safetensors)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the compiled probe NEFF (must NOT overlap the DiT compile dir)",
    )
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=61)
    p.add_argument("--text-seq-len", type=int, default=256)
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    return p.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> int:
    args = _parse_args()
    from difflet.models.hunyuan_video.application import NeuronHunyuanVideoApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    print(f"[probe-compile] source: {args.source_dir}", flush=True)
    print(f"[probe-compile] output: {args.output_dir}", flush=True)
    print(f"[probe-compile] tp_degree: {args.tp_degree}", flush=True)
    print(
        f"[probe-compile] shape: H={args.height} W={args.width} frames={args.num_frames}",
        flush=True,
    )

    t_init = time.perf_counter()
    app = NeuronHunyuanVideoApplication(
        model_path=str(args.source_dir),
        parallel=DiffletParallelConfig(tp_degree=int(args.tp_degree)),
        dtype=_dtype_from_name(args.dtype),
        shape={
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
        },
        text_seq_len=int(args.text_seq_len),
        enable_vae_decoder=False,
    )
    print(
        f"[probe-compile] application instantiated in {time.perf_counter() - t_init:.2f}s",
        flush=True,
    )

    probe_app = app.teacache_probe
    if probe_app is None:
        print("[probe-compile] ERROR: app.teacache_probe is None", file=sys.stderr)
        return 2

    print(
        f"[probe-compile] probe app: {type(probe_app).__name__}; "
        f"probe models: {[m.tag for m in probe_app.models]}",
        flush=True,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe-compile] starting compile to {output_dir}", flush=True)

    t_compile = time.perf_counter()
    try:
        probe_app.compile(str(output_dir))
    except Exception as exc:
        elapsed = time.perf_counter() - t_compile
        print(
            f"[probe-compile] FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
    elapsed = time.perf_counter() - t_compile
    print(f"[probe-compile] compile done in {elapsed:.1f}s", flush=True)

    artifact_files = sorted(output_dir.rglob("*"))
    total_bytes = sum(p.stat().st_size for p in artifact_files if p.is_file())
    print(f"[probe-compile] output dir contains {len(artifact_files)} entries, "
          f"{total_bytes / 1024 / 1024:.1f} MiB total", flush=True)
    for p in artifact_files[:20]:
        kind = "DIR" if p.is_dir() else f"{p.stat().st_size / 1024:.1f} KiB"
        print(f"  {kind:>12}  {p.relative_to(output_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
