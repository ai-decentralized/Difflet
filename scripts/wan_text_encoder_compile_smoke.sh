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
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

MODEL_DIR="${1:-${NOVA_WAN_MODEL_DIR:-/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7}}"
TEXT_SEQ_LEN="${NOVA_WAN_TEXT_SEQ_LEN:-512}"
TP_DEGREE="${NOVA_WAN_TP_DEGREE:-4}"
CP_DEGREE="${NOVA_WAN_CP_DEGREE:-1}"
WORLD_SIZE=$(( TP_DEGREE * CP_DEGREE ))
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-${WORLD_SIZE}}"
OUT_DIR="${NOVA_WAN_TEXT_ENCODER_OUT:-${ROOT}/.nova-cache/wan_text_encoder_smoke}"

cd "${ROOT}"

mkdir -p "${OUT_DIR}"

exec "${PYTHON_BIN}" - <<PY
import os
import torch

from nova.backends.trainium.wan.text_encoder import NeuronWanTextEncoderApplication
from nova.models.wan.application import create_wan_text_encoder_config

model_dir = ${MODEL_DIR@Q}
out_dir = ${OUT_DIR@Q}
text_seq_len = int(${TEXT_SEQ_LEN})
tp_degree = int(${TP_DEGREE})
cp_degree = int(${CP_DEGREE})
world_size = int(${WORLD_SIZE})

print(f"[wan-umt5] model_dir    = {model_dir}")
print(f"[wan-umt5] out_dir      = {out_dir}")
print(f"[wan-umt5] text_seq_len = {text_seq_len}")
print(f"[wan-umt5] tp_degree    = {tp_degree}")
print(f"[wan-umt5] cp_degree    = {cp_degree}")
print(f"[wan-umt5] world_size   = {world_size}")

config = create_wan_text_encoder_config(
    model_path=model_dir,
    world_size=world_size,
    tp_degree=tp_degree,
    dtype=torch.bfloat16,
    text_seq_len=text_seq_len,
    batch_size=1,
)

app = NeuronWanTextEncoderApplication(
    model_path=os.path.join(model_dir, "text_encoder"),
    config=config,
)

app.compile(out_dir, debug=False)
print(f"[wan-umt5] compile completed → {out_dir}")
PY
