#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
NEURON_PYTHON="${NEURON_VENV}/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_PYTHON}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi
if [[ -d "${NEURON_VENV}/bin" ]]; then
  export PATH="${NEURON_VENV}/bin:${PATH}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NOVA_BACKEND="${NOVA_BACKEND:-trainium}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-1}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

MODEL_DIR="${1:-${NOVA_WAN_MODEL_DIR:-/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7}}"
HEIGHT="${NOVA_WAN_HEIGHT:-480}"
WIDTH="${NOVA_WAN_WIDTH:-832}"
FRAMES="${NOVA_WAN_FRAMES:-9}"
OUT_DIR="${NOVA_WAN_VAE_OUT:-${ROOT}/.nova-cache/wan_vae_decoder_smoke}"

cd "${ROOT}"

mkdir -p "${OUT_DIR}"

exec "${PYTHON_BIN}" - <<PY
import os
import torch

from nova.backends.trainium.wan.vae import NeuronWanVAEDecoderApplication
from nova.models.wan.application import create_wan_vae_decoder_config

model_dir = ${MODEL_DIR@Q}
out_dir = ${OUT_DIR@Q}
height = int(${HEIGHT})
width = int(${WIDTH})
frames = int(${FRAMES})

print(f"[wan-vae] model_dir = {model_dir}")
print(f"[wan-vae] out_dir   = {out_dir}")
print(f"[wan-vae] shape     = {(height, width, frames)}")
print("[wan-vae] tp_degree = 1")

config = create_wan_vae_decoder_config(
    model_path=model_dir,
    world_size=1,
    tp_degree=1,
    dtype=torch.bfloat16,
    height=height,
    width=width,
    num_frames=frames,
    batch_size=1,
)

app = NeuronWanVAEDecoderApplication(
    model_path=os.path.join(model_dir, "vae"),
    config=config,
)

app.compile(out_dir, debug=False)
print(f"[wan-vae] compile completed -> {out_dir}")
PY
