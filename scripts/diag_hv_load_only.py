#!/usr/bin/env python3
"""Diagnose whether NeuronHunyuanVideoBackboneApplication.load() against the
existing DiT-only model.pt still works after the probe wrapper was added.

This load goes through application_base.load() which torch.jit.loads the
single model.pt and assigns it to all self.models[].model. The existing
model.pt was traced with only the DiT wrapper. The current code has DiT + probe
in self.models. If the load succeeds, the unified traced_model is forgiving of
the wrapper-count mismatch (probe wrapper just won't have a working backend).
If it fails, the multi-wrapper approach is structurally broken and we need
to split the probe into a separate application.
"""

from __future__ import annotations

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


def main() -> int:
    from nova.models.hunyuan_video.application import NeuronHunyuanVideoApplication
    from nova.pipeline.parallel_config import NovaParallelConfig

    print(f"[diag] uptime check: {time.time()}", flush=True)

    t_init = time.perf_counter()
    app = NeuronHunyuanVideoApplication(
        model_path=".nova-cache/f3_hunyuan_n4_4d8s1r/source",
        parallel=NovaParallelConfig(tp_degree=4),
        dtype=torch.bfloat16,
        shape={"height": 320, "width": 512, "num_frames": 61},
        text_seq_len=256,
        enable_vae_decoder=False,
    )
    print(f"[diag] app instantiated in {time.perf_counter() - t_init:.2f}s", flush=True)
    print(f"[diag] app.transformer = {type(app.transformer).__name__}", flush=True)
    print(
        f"[diag] app.transformer.models = {[m.tag for m in app.transformer.models]}",
        flush=True,
    )

    t_load = time.perf_counter()
    print("[diag] calling app.load(compiled_dir) (existing DiT cache)", flush=True)
    try:
        app.load(".nova-cache/f3_hunyuan_n4_4d8s1r/compiled", skip_warmup=True)
        elapsed = time.perf_counter() - t_load
        print(f"[diag] load SUCCEEDED in {elapsed:.1f}s", flush=True)
    except Exception as exc:
        elapsed = time.perf_counter() - t_load
        print(
            f"[diag] load FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise

    # If load succeeded, probe whether the wrappers are usable
    print(
        f"[diag] transformer.models[0].model is None? "
        f"{app.transformer.models[0].model is None}",
        flush=True,
    )
    print(
        f"[diag] transformer.models[1].model is None? "
        f"{app.transformer.models[1].model is None}",
        flush=True,
    )
    print(
        f"[diag] models[0].model is models[1].model? "
        f"{app.transformer.models[0].model is app.transformer.models[1].model}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
