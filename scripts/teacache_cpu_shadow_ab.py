#!/usr/bin/env python3
"""A/B test: CPU shadow vs Trainium probe NEFF for TeaCache mod_input.

Loads both paths against the same HV bundle and compares:
1. Numerical parity — cosine between CPU-shadow mod_input and probe-NEFF mod_input
2. Per-call latency — CPU shadow ms/call vs probe NEFF ms/call

cclog 77 hypothesis: CPU shadow is ~5-15 ms/call (no NEFF dispatch / mark_step)
vs probe NEFF ~51 ms/call.
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
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402

SOURCE = ROOT / ".difflet-cache" / "f3_hunyuan_n4_4d8s1r" / "source"
COMPILED = ROOT / ".difflet-cache" / "f3_hunyuan_n4_4d8s1r" / "compiled"
BUNDLE = ROOT / ".difflet-cache" / "hunyuan_dit_inputs" / "cat_walking_4step.safetensors"
META = Path(str(BUNDLE) + ".meta.json")
OUT = ROOT / "cclogs" / "m9-teacache" / "cpu_shadow_ab.json"


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            a.detach().float().cpu().reshape(1, -1),
            b.detach().float().cpu().reshape(1, -1),
            dim=1,
        ).item()
    )


def main() -> int:
    from difflet.backends.trainium.hunyuan_video.teacache_cpu_shadow import (
        HunyuanVideoTeacacheCPUShadow,
    )
    from difflet.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    meta = json.loads(META.read_text())
    tensors = {
        k: v.to(dtype=torch.bfloat16) if v.is_floating_point() else v
        for k, v in load_safetensors_file(str(BUNDLE), device="cpu").items()
    }
    inputs = (
        tensors["latents_init"],
        tensors["timesteps"][:1].clone(),
        tensors["encoder_hidden_states"],
        tensors["encoder_attention_mask"],
        tensors["pooled_projections"],
        tensors["guidance"],
    )

    # ---- CPU shadow ----
    print("[ab] loading CPU shadow...", flush=True)
    t = time.perf_counter()
    shadow = HunyuanVideoTeacacheCPUShadow(str(SOURCE), dtype=torch.bfloat16)
    print(f"[ab] CPU shadow loaded in {time.perf_counter() - t:.1f}s", flush=True)

    # warm + time CPU shadow (10 calls)
    cpu_mod = shadow.teacache_mod_input(*inputs)
    cpu_times = []
    for _ in range(10):
        t = time.perf_counter()
        cpu_mod = shadow.teacache_mod_input(*inputs)
        cpu_times.append((time.perf_counter() - t) * 1000.0)
    cpu_ms = sorted(cpu_times)[len(cpu_times) // 2]  # median
    print(f"[ab] CPU shadow median = {cpu_ms:.1f} ms/call", flush=True)

    # ---- Trainium probe NEFF ----
    print("[ab] loading Trainium app + probe NEFF...", flush=True)
    app = NeuronHunyuanVideoApplication(
        model_path=str(SOURCE),
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype=torch.bfloat16,
        shape={"height": int(meta["height"]), "width": int(meta["width"]),
               "num_frames": int(meta["num_frames"])},
        text_seq_len=int(meta["text_seq_len"]),
        enable_vae_decoder=False,
    )
    app.load(str(COMPILED), skip_warmup=True)
    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=inputs[0], timestep=inputs[1],
        encoder_hidden_states=inputs[2], encoder_attention_mask=inputs[3],
        pooled_projections=inputs[4], guidance=inputs[5],
    )
    probe_mod = app.teacache_mod_input(bundle)  # warm
    probe_times = []
    for _ in range(10):
        t = time.perf_counter()
        probe_mod = app.teacache_mod_input(bundle)
        _ = probe_mod.detach().cpu()  # force materialize like real use
        probe_times.append((time.perf_counter() - t) * 1000.0)
    probe_ms = sorted(probe_times)[len(probe_times) // 2]
    print(f"[ab] probe NEFF median = {probe_ms:.1f} ms/call", flush=True)

    cosine = _cosine(cpu_mod, probe_mod)
    print(f"[ab] mod_input cosine (CPU vs NEFF) = {cosine:.6f}", flush=True)

    result = {
        "schema": "difflet-m9-teacache-cpu-shadow-ab-v1",
        "model": "hunyuan_video",
        "shape_label": f"{meta['height']}x{meta['width']}x{meta['num_frames']}",
        "cpu_shadow_ms_per_call": cpu_ms,
        "probe_neff_ms_per_call": probe_ms,
        "speedup_ratio_probe_over_cpu": probe_ms / cpu_ms if cpu_ms else None,
        "mod_input_cosine_cpu_vs_neff": cosine,
        "cpu_call_times_ms": cpu_times,
        "probe_call_times_ms": probe_times,
        "hardware_measured": True,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[ab] wrote {OUT}", flush=True)
    print(
        f"[ab] SUMMARY: CPU {cpu_ms:.1f} ms vs NEFF {probe_ms:.1f} ms "
        f"({probe_ms / cpu_ms:.1f}x faster) | cosine {cosine:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
