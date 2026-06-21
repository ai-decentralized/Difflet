#!/usr/bin/env python3
"""Isolate the HV-1.5 monolithic NaN: is it the DiT base path or the denoise loop?

Loads the (cached) HV-1.5 DiT + probe, runs ONE forward on the bundle's REAL step-0
input (timestep=1000, real embeds), and reports whether DiT noise_pred and probe delta
are finite. Decides whether the NaN is in the base DiT path (inputs/shape/weights) or
introduced by the multi-step schedule.
"""

from __future__ import annotations

import os
import sys
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

MODEL_DIR = "/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-720p_t2v/snapshots/f4dbc4a1efa4ac8ea56680cdf79d9f455105e814"
BUNDLE = ROOT / ".difflet-cache" / "hunyuan15_dit_inputs" / "real_320x512x61_4step.safetensors"
COMPILED = ROOT / ".difflet-cache" / "hv15_teacache" / "compiled"


def _fin(x):
    x = x.detach().float()
    return f"finite={bool(torch.isfinite(x).all())} nan={int(torch.isnan(x).sum())} min={x.min():.3g} max={x.max():.3g}"


def main() -> int:
    from difflet.models.hunyuan_video.application import (
        HunyuanVideo15DiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    tns = load_safetensors_file(str(BUNDLE), device="cpu")
    dtype = torch.bfloat16
    print("[iso] input checks:", flush=True)
    for k in ("hidden_states", "encoder_hidden_states", "encoder_hidden_states_2", "image_embeds"):
        print(f"  {k}: {_fin(tns[k])}", flush=True)

    app = NeuronHunyuanVideoApplication(
        model_path=MODEL_DIR,
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype=dtype,
        shape={"height": 320, "width": 512, "num_frames": 61},
        model_version="1.5",
        transformer_runtime="monolithic",
        text_seq_len=int(tns["encoder_hidden_states"].shape[1]),
        enable_vae_decoder=False,
        teacache_fused=True,
    )
    app.load(str(COMPILED), skip_warmup=True)
    print("[iso] loaded", flush=True)

    # cclog 86: the NaN is the attention softmax on fully-masked query rows (the
    # symmetric mask over padding tokens -> all-(-inf) row -> 0/0 on the Neuron
    # kernel; CPU SDPA handles it). Relaxing BOTH masks to all-valid removes the
    # all-masked rows -> should be FINITE on Trainium, confirming the cause.
    relax = os.environ.get("HV15_RELAX_MASKS") == "1"

    def bundle_at(ts_val):
        eam = tns["encoder_attention_mask"].to(torch.int64)
        eam2 = tns["encoder_attention_mask_2"].to(torch.int64)
        if relax:
            eam = torch.ones_like(eam)
            eam2 = torch.ones_like(eam2)
        return HunyuanVideo15DiTInputBundle(
            hidden_states=tns["hidden_states"].to(dtype),
            timestep=torch.tensor([ts_val], dtype=dtype),
            encoder_hidden_states=tns["encoder_hidden_states"].to(dtype),
            encoder_attention_mask=eam,
            timestep_r=tns["timestep_r"].to(dtype),
            encoder_hidden_states_2=tns["encoder_hidden_states_2"].to(dtype),
            encoder_attention_mask_2=eam2,
            image_embeds=tns["image_embeds"].to(dtype),
        )

    for ts_val in (1000.0, 500.0, 50.0):
        b = bundle_at(ts_val)
        out = app.forward_dit(b)
        out = out[0] if isinstance(out, (tuple, list)) else out
        print(f"[iso] DiT  t={ts_val}: {_fin(out)}", flush=True)
        d = app.teacache_probe.teacache_delta(b.hidden_states, b.timestep, b.timestep_r)
        d = d.detach().float().reshape(-1)[0]
        print(f"[iso] probe t={ts_val}: delta={d.item():.6g} finite={bool(torch.isfinite(d))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
