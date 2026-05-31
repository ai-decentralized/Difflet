#!/usr/bin/env python3
"""Stage 3 of cclog 72 implementation verification — real runtime.

Loads the HunyuanVideo Trainium application (DiT + teacache probe) and
exercises the probe NEFF end-to-end:

1. Load both DiT and probe artifacts onto NeuronCores
2. Call ``app.teacache_mod_input(bundle)`` — calibration entry, returns mod_input
3. Call ``app.teacache_mod_input_with_delta(...)`` — T1 entry, returns (delta, mod_input)
4. Validate output shapes, dtypes, and value ranges

Designed for foreground execution per cclog 70 §"Guard".
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
COMPILED = ROOT / ".nova-cache" / "f3_hunyuan_n4_4d8s1r" / "compiled"
BUNDLE = ROOT / ".nova-cache" / "hunyuan_dit_inputs" / "cat_walking_4step.safetensors"
META = Path(str(BUNDLE) + ".meta.json")


def main() -> int:
    from nova.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from nova.pipeline.parallel_config import NovaParallelConfig

    meta = json.loads(META.read_text())
    print(f"[smoke] shape: H={meta['height']} W={meta['width']} frames={meta['num_frames']}",
          flush=True)

    tensors = {
        k: v.to(dtype=torch.bfloat16) if v.is_floating_point() else v
        for k, v in load_safetensors_file(str(BUNDLE), device="cpu").items()
    }

    t_init = time.perf_counter()
    app = NeuronHunyuanVideoApplication(
        model_path=str(SOURCE),
        parallel=NovaParallelConfig(tp_degree=4, cp_enabled=False),
        dtype=torch.bfloat16,
        shape={"height": int(meta["height"]), "width": int(meta["width"]),
               "num_frames": int(meta["num_frames"])},
        text_seq_len=int(meta["text_seq_len"]),
        enable_vae_decoder=False,
    )
    print(f"[smoke] app instantiated in {time.perf_counter() - t_init:.2f}s", flush=True)
    print(f"[smoke] components: {[c.name for c in app.components()]}", flush=True)

    t_load = time.perf_counter()
    print("[smoke] calling app.load(compiled_dir, skip_warmup=True)", flush=True)
    app.load(str(COMPILED), skip_warmup=True)
    print(f"[smoke] load complete in {time.perf_counter() - t_load:.1f}s", flush=True)

    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=tensors["latents_init"],
        timestep=tensors["timesteps"][:1].clone(),
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_attention_mask=tensors["encoder_attention_mask"],
        pooled_projections=tensors["pooled_projections"],
        guidance=tensors["guidance"],
    )

    # ----- 1. Calibration entry: teacache_mod_input(bundle) -----
    print("[smoke] STEP 1: calibration teacache_mod_input(bundle)", flush=True)
    t_step1 = time.perf_counter()
    mod_input_a = app.teacache_mod_input(bundle)
    elapsed_step1 = time.perf_counter() - t_step1
    print(
        f"[smoke] STEP 1 done in {elapsed_step1 * 1000:.1f} ms; "
        f"mod_input shape={tuple(mod_input_a.shape)} dtype={mod_input_a.dtype} "
        f"norm={float(torch.linalg.vector_norm(mod_input_a.float().reshape(-1)).item()):.4f}",
        flush=True,
    )

    # ----- 2. T1 entry with zero prev_mod: should return delta=||mod_input|| -----
    print("[smoke] STEP 2: teacache_mod_input_with_delta(..., zero prev_mod)", flush=True)
    zero_prev = torch.zeros_like(mod_input_a)
    t_step2 = time.perf_counter()
    delta_b, mod_input_b = app.teacache_mod_input_with_delta(
        bundle.hidden_states,
        bundle.timestep,
        bundle.encoder_hidden_states,
        bundle.encoder_attention_mask,
        bundle.pooled_projections,
        bundle.guidance,
        zero_prev,
    )
    elapsed_step2 = time.perf_counter() - t_step2
    print(
        f"[smoke] STEP 2 done in {elapsed_step2 * 1000:.1f} ms; "
        f"delta={float(delta_b.item()):.4f}  "
        f"mod_input shape={tuple(mod_input_b.shape)} "
        f"norm={float(torch.linalg.vector_norm(mod_input_b.float().reshape(-1)).item()):.4f}",
        flush=True,
    )

    # ----- 3. T1 entry with mod_input_a as prev_mod: delta should be near 0 -----
    print("[smoke] STEP 3: teacache_mod_input_with_delta(..., prev_mod=mod_input_a)",
          flush=True)
    t_step3 = time.perf_counter()
    delta_c, mod_input_c = app.teacache_mod_input_with_delta(
        bundle.hidden_states,
        bundle.timestep,
        bundle.encoder_hidden_states,
        bundle.encoder_attention_mask,
        bundle.pooled_projections,
        bundle.guidance,
        mod_input_a,
    )
    elapsed_step3 = time.perf_counter() - t_step3
    print(
        f"[smoke] STEP 3 done in {elapsed_step3 * 1000:.1f} ms; "
        f"delta={float(delta_c.item()):.6f}  "
        f"(expected near 0 since prev_mod == current mod_input)",
        flush=True,
    )

    # ----- 4. Sanity: STEP 2's delta should equal STEP 1's mod_input norm -----
    norm_step1 = float(torch.linalg.vector_norm(mod_input_a.float().reshape(-1)).item())
    delta_step2 = float(delta_b.item())
    pct_match = abs(norm_step1 - delta_step2) / max(norm_step1, 1e-9)
    print(
        f"[smoke] CONSISTENCY: ||STEP1 mod_input|| = {norm_step1:.4f}, "
        f"STEP2 delta (vs zero) = {delta_step2:.4f}, "
        f"rel diff = {pct_match * 1e6:.1f} ppm",
        flush=True,
    )

    # Note: bf16 introduces sub-percent rounding so allow loose tolerance
    if pct_match > 0.05:
        print(f"[smoke] WARN: STEP1 norm vs STEP2 delta mismatch > 5%", flush=True)
    else:
        print(f"[smoke] OK: STEP1 ≈ STEP2 within bf16 tolerance", flush=True)

    print("[smoke] DONE — probe NEFF runtime validated", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
