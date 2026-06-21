#!/usr/bin/env python3
"""cclog 86 diagnostic: is the monolithic HV-1.5 DiT NaN size-dependent?

Compiles the HV-1.5 DiT at SMALL frame counts (short image-token sequence) and runs
one forward each, checking for NaN. Compares to the known NaN at num_frames=61
(seq 10240). If a tiny shape is FINITE -> the NaN scales with size (memory / a
length-dependent numerical path) and a smaller shape is usable. If still NaN at tiny
shape -> the NaN is size-independent (a precision bug everywhere).

DiT only (teacache_fused=False) for fast compiles. Random latent of typical magnitude
(~N(0,1)) + cached shape-independent text/image embeds (mask_2 relaxed to all-valid).
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

import time  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402

MODEL_DIR = "/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-720p_t2v/snapshots/f4dbc4a1efa4ac8ea56680cdf79d9f455105e814"
BUNDLE = ROOT / ".difflet-cache" / "hunyuan15_dit_inputs" / "real_320x512x61_4step.safetensors"
FRAMES = [int(x) for x in os.environ.get("HV15_FRAMES", "1,5").split(",")]


def _fin(x):
    x = x.detach().float()
    return bool(torch.isfinite(x).all()), int(torch.isnan(x).sum())


def main() -> int:
    from difflet.models.hunyuan_video.application import (
        HunyuanVideo15DiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    tns = load_safetensors_file(str(BUNDLE), device="cpu")
    dtype = torch.bfloat16
    mask2 = tns["encoder_attention_mask_2"].to(torch.int64)
    if int(mask2.sum()) == 0:
        mask2 = torch.ones_like(mask2)
    ehs = tns["encoder_hidden_states"].to(dtype)
    eam = tns["encoder_attention_mask"].to(torch.int64)
    ehs2 = tns["encoder_hidden_states_2"].to(dtype)
    img = tns["image_embeds"].to(dtype)
    tr = tns["timestep_r"].to(dtype)

    for nf in FRAMES:
        print(f"\n[sweep] === num_frames={nf} ===", flush=True)
        app = NeuronHunyuanVideoApplication(
            model_path=MODEL_DIR,
            parallel=DiffletParallelConfig(tp_degree=4),
            dtype=dtype,
            shape={"height": 320, "width": 512, "num_frames": nf},
            model_version="1.5",
            transformer_runtime="monolithic",
            text_seq_len=int(ehs.shape[1]),
            enable_vae_decoder=False,
            teacache_fused=False,
        )
        cfg = app.transformer.config
        lf, lh, lw = int(cfg.latent_frames), int(cfg.latent_height), int(cfg.latent_width)
        seq = lf * lh * lw
        print(f"[sweep] latent ({lf},{lh},{lw}) image_seq={seq} total≈{seq + ehs.shape[1] + ehs2.shape[1]}",
              flush=True)
        cdir = ROOT / ".difflet-cache" / f"hv15_nan_sweep_f{nf}"
        t = time.time()
        app.compile(str(cdir))
        app.load(str(cdir), skip_warmup=True)
        print(f"[sweep] compiled+loaded in {time.time()-t:.0f}s", flush=True)

        torch.manual_seed(0)
        latent = torch.randn(1, 32, lf, lh, lw, dtype=dtype)
        cond = torch.zeros(1, 33, lf, lh, lw, dtype=dtype)
        for ts_val in (1000.0, 500.0):
            b = HunyuanVideo15DiTInputBundle(
                hidden_states=torch.cat([latent, cond], dim=1),
                timestep=torch.tensor([ts_val], dtype=dtype),
                encoder_hidden_states=ehs,
                encoder_attention_mask=eam,
                timestep_r=tr,
                encoder_hidden_states_2=ehs2,
                encoder_attention_mask_2=mask2,
                image_embeds=img,
            )
            out = app.forward_dit(b)
            out = out[0] if isinstance(out, (tuple, list)) else out
            fin, nans = _fin(out)
            print(f"[sweep] num_frames={nf} seq={seq} t={ts_val}: FINITE={fin} nan={nans}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
