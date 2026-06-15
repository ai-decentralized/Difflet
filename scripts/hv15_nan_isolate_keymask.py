#!/usr/bin/env python3
"""Dual-mask NaN isolation on the KEYMASK (option-A) HV-1.5 NEFF.

Loads the hv15_teacache_keymask compiled artifact (static-reorder forward + key-mask
attention + key-only refiner) and runs ONE forward with BOTH real and relaxed masks.
If real=NaN but relaxed=finite -> the NaN is STILL mask-dependent (a fix is bypassed).
If both NaN -> mask-independent (e.g. a bf16-range issue, not the mask path).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]

# monolithic-fix gate: static forward + key-mask main blocks + host refiner (cclog 86).
os.environ["NOVA_HUNYUAN15_KEY_MASK_ATTENTION"] = "1"
os.environ["NOVA_HUNYUAN15_HOST_REFINER"] = "1"


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
BUNDLE = ROOT / ".nova-cache" / "hunyuan15_dit_inputs" / "real_320x512x61_4step.safetensors"
COMPILED = ROOT / ".nova-cache" / "hv15_teacache_keymask" / "compiled"


def _fin(x):
    x = x.detach().float()
    return f"finite={bool(torch.isfinite(x).all())} nan={int(torch.isnan(x).sum())} min={x.min():.3g} max={x.max():.3g}"


def main() -> int:
    from nova.models.hunyuan_video.application import (
        HunyuanVideo15DiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from nova.pipeline.parallel_config import NovaParallelConfig

    tns = load_safetensors_file(str(BUNDLE), device="cpu")
    dtype = torch.bfloat16

    app = NeuronHunyuanVideoApplication(
        model_path=MODEL_DIR,
        parallel=NovaParallelConfig(tp_degree=4),
        dtype=dtype,
        shape={"height": 320, "width": 512, "num_frames": 61},
        model_version="1.5",
        transformer_runtime="monolithic",
        text_seq_len=int(tns["encoder_hidden_states"].shape[1]),
        enable_vae_decoder=False,
        teacache_fused=True,
    )
    app.load(str(COMPILED), skip_warmup=True)
    print("[iso-km] loaded", flush=True)

    def bundle_at(ts_val, relax):
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

    for relax in (True, False):
        b = bundle_at(1000.0, relax)
        out = app.forward_dit(b)
        out = out[0] if isinstance(out, (tuple, list)) else out
        tag = "relaxed(all-ones)" if relax else "real"
        print(f"[iso-km] DiT t=1000 mask={tag}: {_fin(out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
