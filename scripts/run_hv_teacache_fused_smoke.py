#!/usr/bin/env python3
"""cclog 80 Step 3: compile + run the fused-A probe on HV N4 4d8s1r.

Verifies:
  1. compiles (alias on the prev_mod Parameter)
  2. loads + runs; prev_mod persists/updates on device across calls (delta on
     repeated identical input → 0 after first, since prev_mod becomes mod_input)
  3. per-call latency ≈ 26 ms (vs current probe 52 ms) — cclog 79 Test 1 target
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

SOURCE = ROOT / ".nova-cache" / "f3_hunyuan_n4_4d8s1r" / "source"
OUT_DIR = ROOT / ".nova-cache" / "f3_hunyuan_n4_4d8s1r" / "compiled" / "teacache_probe_fused"
BUNDLE = ROOT / ".nova-cache" / "hunyuan_dit_inputs" / "cat_walking_4step.safetensors"
META = Path(str(BUNDLE) + ".meta.json")


def main() -> int:
    from nova.backends.trainium.hunyuan_video.teacache_probe import (
        NeuronHunyuanVideoTeacacheProbeFusedApplication,
    )
    from nova.models.hunyuan_video.application import create_hunyuan_video_backbone_config

    meta = json.loads(META.read_text())
    tensors = {
        k: v.to(dtype=torch.bfloat16) if v.is_floating_point() else v
        for k, v in load_safetensors_file(str(BUNDLE), device="cpu").items()
    }
    args6 = (
        tensors["latents_init"],
        tensors["timesteps"][:1].clone(),
        tensors["encoder_hidden_states"],
        tensors["encoder_attention_mask"],
        tensors["pooled_projections"],
        tensors["guidance"],
    )

    config = create_hunyuan_video_backbone_config(
        model_path=str(SOURCE), world_size=4, tp_degree=4, dtype=torch.bfloat16,
        height=int(meta["height"]), width=int(meta["width"]),
        num_frames=int(meta["num_frames"]), text_seq_len=int(meta["text_seq_len"]),
        batch_size=1,
    )
    app = NeuronHunyuanVideoTeacacheProbeFusedApplication(
        model_path=str(SOURCE), config=config
    )
    print("[fused] compiling...", flush=True)
    t = time.perf_counter()
    app.compile(str(OUT_DIR))
    print(f"[fused] compiled in {time.perf_counter() - t:.1f}s", flush=True)
    app.load(str(OUT_DIR), skip_warmup=True)
    print("[fused] loaded", flush=True)

    # warm
    d0 = app.teacache_delta(*args6)
    v0 = float(d0.detach().cpu().reshape(-1)[0].item())
    print(f"[fused] call 1 delta = {v0:.4f}  (vs zero prev_mod = ||mod_input||)", flush=True)

    # second call with SAME input: prev_mod is now mod_input → delta should be ~0
    d1 = app.teacache_delta(*args6)
    v1 = float(d1.detach().cpu().reshape(-1)[0].item())
    print(f"[fused] call 2 delta = {v1:.6f}  (same input → expect ~0 if prev_mod persisted)",
          flush=True)

    persisted = v1 < v0 * 1e-3  # delta collapsed → prev_mod tracked on device
    print(f"[fused] PREV_MOD_PERSISTS_ON_DEVICE = {persisted}", flush=True)

    # timing
    times = []
    for _ in range(20):
        t = time.perf_counter()
        out = app.teacache_delta(*args6)
        _ = float(out.detach().cpu().reshape(-1)[0].item())
        times.append((time.perf_counter() - t) * 1000.0)
    median_ms = sorted(times)[len(times) // 2]
    print(f"[fused] median per-call = {median_ms:.2f} ms  (current probe = ~52 ms)", flush=True)

    result = {
        "schema": "nova-m9-teacache-fused-smoke-v1",
        "call1_delta": v0, "call2_delta_same_input": v1,
        "prev_mod_persists": persisted,
        "median_ms_per_call": median_ms,
        "current_probe_ms": 52.25,
        "hardware_measured": True,
    }
    out_path = ROOT / "cclogs" / "m9-teacache" / "fused_probe_smoke.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[fused] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
