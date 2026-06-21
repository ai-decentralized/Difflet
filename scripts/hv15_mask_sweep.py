#!/usr/bin/env python3
"""Free (no-recompile) runtime sweep to localize the HV-1.5 real-mask NaN.

Same keymask NEFF, vary ONLY the runtime mask values:
  - relax mllm only / relax byt5 only  -> which stream's mask triggers it
  - sweep mllm valid-count [1000,512,128,64,13,1] -> count-dependent or structural
The NEFF uses runtime mask values (traced with all-ones), so this needs no recompile.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]
os.environ["DIFFLET_HUNYUAN15_KEY_MASK_ATTENTION"] = "1"


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
COMPILED = ROOT / ".difflet-cache" / "hv15_teacache_keymask" / "compiled"


def main() -> int:
    from difflet.models.hunyuan_video.application import (
        HunyuanVideo15DiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    tns = load_safetensors_file(str(BUNDLE), device="cpu")
    dtype = torch.bfloat16
    real_eam = tns["encoder_attention_mask"].to(torch.int64)
    real_eam2 = tns["encoder_attention_mask_2"].to(torch.int64)
    n1 = real_eam.shape[1]
    n2 = real_eam2.shape[1]
    print(f"[sweep] real mllm valid={int(real_eam.sum())}/{n1}  byt5 valid={int(real_eam2.sum())}/{n2}", flush=True)

    app = NeuronHunyuanVideoApplication(
        model_path=MODEL_DIR,
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype=dtype,
        shape={"height": 320, "width": 512, "num_frames": 61},
        model_version="1.5",
        transformer_runtime="monolithic",
        text_seq_len=n1,
        enable_vae_decoder=False,
        teacache_fused=True,
    )
    app.load(str(COMPILED), skip_warmup=True)
    print("[sweep] loaded", flush=True)

    def run(eam, eam2, tag):
        b = HunyuanVideo15DiTInputBundle(
            hidden_states=tns["hidden_states"].to(dtype),
            timestep=torch.tensor([1000.0], dtype=dtype),
            encoder_hidden_states=tns["encoder_hidden_states"].to(dtype),
            encoder_attention_mask=eam,
            timestep_r=tns["timestep_r"].to(dtype),
            encoder_hidden_states_2=tns["encoder_hidden_states_2"].to(dtype),
            encoder_attention_mask_2=eam2,
            image_embeds=tns["image_embeds"].to(dtype),
        )
        out = app.forward_dit(b)
        out = out[0] if isinstance(out, (tuple, list)) else out
        fin = bool(torch.isfinite(out.float()).all())
        print(f"[sweep] {tag:32s}: finite={fin}", flush=True)

    ones1 = torch.ones_like(real_eam)
    ones2 = torch.ones_like(real_eam2)
    run(ones1, ones2, "both ones")
    run(real_eam, ones2, "mllm real, byt5 ones")
    run(ones1, real_eam2, "mllm ones, byt5 real")
    run(real_eam, real_eam2, "both real")

    # mllm valid-count sweep (byt5 = ones to isolate)
    for k in (512, 128, 64, 13, 1):
        m = torch.zeros_like(real_eam)
        m[:, :k] = 1
        run(m, ones2, f"mllm first-{k} valid, byt5 ones")

    # multiple-of-128 hypothesis: multiples finite, non-multiples NaN
    for k in (256, 384, 200, 300, 129, 192):
        m = torch.zeros_like(real_eam)
        m[:, :k] = 1
        run(m, ones2, f"mllm first-{k} (mult128={k % 128 == 0})")

    # where are the REAL valid tokens? (contiguous prefix vs scattered)
    idx = torch.nonzero(real_eam[0]).flatten().tolist()
    print(f"[sweep] real mllm valid indices: {idx}", flush=True)

    # round real valid region UP to a multiple of 128 (mark extra padding valid)
    real_k = int(real_eam.sum())
    padded_k = ((real_k + 127) // 128) * 128
    m = torch.zeros_like(real_eam)
    m[:, :padded_k] = 1
    run(m, real_eam2, f"mllm round-up first-{padded_k}, byt5 real")
    return 0


if __name__ == "__main__":
    sys.exit(main())
