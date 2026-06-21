#!/usr/bin/env python3
"""cclog 81: compile + per-call smoke for the Qwen-Image fused-A probe.

Verifies the fused recipe generalizes to Qwen: compiles the fused probe (alias
on prev_mod), loads, checks prev_mod persists on device (delta -> 0 on repeat),
and times per-call. Analog of run_hv_teacache_fused_smoke.py (HV got 8.85 ms).
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

import torch  # noqa: E402
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402

BUNDLE = ROOT / ".difflet-cache" / "qwen_image_dit_inputs" / "full_1024_4step.safetensors"
META = Path(str(BUNDLE) + ".meta.json")
OUT_DIR = ROOT / ".difflet-cache" / "qwen_fused_probe"


def main() -> int:
    from difflet.backends.trainium.qwen_image.teacache_probe_fused import (
        NeuronQwenImageTeacacheProbeFusedApplication,
    )
    from difflet.models.qwen_image.application import create_qwen_image_transformer_config

    meta = json.loads(META.read_text())
    src = meta["model_id"]
    print(f"[qwen-fused] source: {src}", flush=True)
    tensors = {
        k: v.to(dtype=torch.bfloat16) if v.is_floating_point() else v
        for k, v in load_safetensors_file(str(BUNDLE), device="cpu").items()
    }
    args5 = (
        tensors["latents_init"],
        tensors["timesteps"][:1].clone().to(torch.bfloat16),
        tensors["encoder_hidden_states"],
        tensors["encoder_hidden_states_mask"],
        tensors["guidance"].to(torch.bfloat16),
    )

    config = create_qwen_image_transformer_config(
        model_path=src, world_size=4, tp_degree=4, dtype=torch.bfloat16,
        height=int(meta["height"]), width=int(meta["width"]),
        text_seq_len=int(meta["text_seq_len"]),
    )
    app = NeuronQwenImageTeacacheProbeFusedApplication(model_path=os.path.join(src, "transformer"), config=config)
    print("[qwen-fused] compiling fused probe...", flush=True)
    t = time.perf_counter()
    app.compile(str(OUT_DIR))
    print(f"[qwen-fused] compiled in {time.perf_counter() - t:.1f}s", flush=True)
    app.load(str(OUT_DIR), skip_warmup=True)
    print("[qwen-fused] loaded", flush=True)

    d0 = float(app.teacache_delta(*args5).detach().cpu().reshape(-1)[0].item())
    print(f"[qwen-fused] call 1 delta = {d0:.4f}", flush=True)
    d1 = float(app.teacache_delta(*args5).detach().cpu().reshape(-1)[0].item())
    print(f"[qwen-fused] call 2 delta = {d1:.6f} (same input -> ~0 if prev_mod persisted)",
          flush=True)
    persisted = d1 < d0 * 1e-3
    print(f"[qwen-fused] PREV_MOD_PERSISTS = {persisted}", flush=True)

    times = []
    for _ in range(20):
        t = time.perf_counter()
        _ = float(app.teacache_delta(*args5).detach().cpu().reshape(-1)[0].item())
        times.append((time.perf_counter() - t) * 1000.0)
    med = sorted(times)[len(times) // 2]
    print(f"[qwen-fused] median per-call = {med:.2f} ms", flush=True)

    res = {
        "schema": "difflet-m9-qwen-fused-smoke-v1",
        "call1_delta": d0, "call2_delta_same_input": d1,
        "prev_mod_persists": persisted, "median_ms_per_call": med,
        "hardware_measured": True,
    }
    op = ROOT / "cclogs" / "m9-teacache" / "qwen_fused_smoke.json"
    op.write_text(json.dumps(res, indent=2) + "\n")
    print(f"[qwen-fused] wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
