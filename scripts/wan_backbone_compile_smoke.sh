#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
NEURON_PYTHON="${NEURON_VENV}/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_PYTHON}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
if [[ -d "${NEURON_VENV}/bin" ]]; then
  export PATH="${NEURON_VENV}/bin:${PATH}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NOVA_BACKEND="${NOVA_BACKEND:-trainium}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

MODEL="${1:-${NOVA_WAN_MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}}"
HEIGHT="${NOVA_WAN_HEIGHT:-480}"
WIDTH="${NOVA_WAN_WIDTH:-832}"
VIDEO_FRAMES="${NOVA_WAN_FRAMES:-9}"
TP_DEGREE="${NOVA_WAN_TP_DEGREE:-4}"
CP_DEGREE="${NOVA_WAN_CP_DEGREE:-1}"
WORLD_SIZE=$(( TP_DEGREE * CP_DEGREE ))
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-${WORLD_SIZE}}"
SUBFOLDER="${NOVA_WAN_TRANSFORMER_SUBFOLDER:-transformer}"
LOCAL_FILES_ONLY="${NOVA_LOCAL_FILES_ONLY:-0}"

if [[ "${SUBFOLDER}" == "transformer_2" ]]; then
  DEFAULT_OUT="${ROOT}/.nova-cache/wan_backbone_2_smoke"
else
  DEFAULT_OUT="${ROOT}/.nova-cache/wan_backbone_smoke"
fi
OUT_DIR="${NOVA_WAN_BACKBONE_OUT:-${DEFAULT_OUT}}"

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

exec "${PYTHON_BIN}" - <<PY
import os
import time
from pathlib import Path

import torch

from nova.backends.trainium.wan.backbone import NeuronWanBackboneApplication
from nova.models.wan.application import create_wan_backbone_config, _latent_num_frames
from nova.pipeline.path_resolver import resolve_model_path

model = ${MODEL@Q}
out_dir = ${OUT_DIR@Q}
height = int(${HEIGHT})
width = int(${WIDTH})
video_frames = int(${VIDEO_FRAMES})
latent_frames = _latent_num_frames(video_frames)
tp_degree = int(${TP_DEGREE})
cp_degree = int(${CP_DEGREE})
world_size = int(${WORLD_SIZE})
subfolder = ${SUBFOLDER@Q}
local_files_only = ${LOCAL_FILES_ONLY@Q} == "1"

model_dir = resolve_model_path(model, local_files_only=local_files_only)
component_dir = os.path.join(model_dir, subfolder)
if not os.path.exists(os.path.join(component_dir, "config.json")):
    raise FileNotFoundError(f"missing {subfolder}/config.json under {model_dir}")

print(f"[wan-backbone] model_dir      = {model_dir}")
print(f"[wan-backbone] subfolder      = {subfolder}")
print(f"[wan-backbone] out_dir        = {out_dir}")
print(f"[wan-backbone] video shape    = ({height}, {width}, {video_frames})")
print(f"[wan-backbone] latent frames  = {latent_frames}")
print(f"[wan-backbone] tp_degree      = {tp_degree}")
print(f"[wan-backbone] cp_degree      = {cp_degree}")
print(f"[wan-backbone] world_size     = {world_size}")

config = create_wan_backbone_config(
    model_path=model_dir,
    world_size=world_size,
    tp_degree=tp_degree,
    dtype=torch.bfloat16,
    height=height,
    width=width,
    num_frames=latent_frames,
    batch_size=1,
    subfolder=subfolder,
    context_parallel_enabled=cp_degree > 1,
)
app = NeuronWanBackboneApplication(model_path=component_dir, config=config)
start = time.time()
app.compile(out_dir, debug=False)
elapsed = time.time() - start
print(f"[wan-backbone] compile completed -> {out_dir}")
print(f"[wan-backbone] elapsed = {elapsed:.3f}s")
PY
