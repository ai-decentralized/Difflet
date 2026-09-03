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
MODEL_DIR="${1:-${DIFFLET_HUNYUAN_MODEL_DIR:-/home/ubuntu/.cache/huggingface/hub/hunyuanvideo-real}}"
HEIGHT="${DIFFLET_HUNYUAN_HEIGHT:-320}"
WIDTH="${DIFFLET_HUNYUAN_WIDTH:-512}"
FRAMES="${DIFFLET_HUNYUAN_FRAMES:-61}"
OUT_DIR="${DIFFLET_HUNYUAN_VAE_OUT:-${ROOT}/.difflet-cache/hunyuan_vae_decoder_smoke}"
WORLD_SIZE="${DIFFLET_HUNYUAN_VAE_WORLD_SIZE:-1}"

export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-${WORLD_SIZE}}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

exec "${PYTHON_BIN}" - <<PY
import os
import time

import torch

from difflet.models.hunyuan_video.application import NeuronHunyuanVideoApplication
from difflet.pipeline.parallel_config import DiffletParallelConfig

model_dir = ${MODEL_DIR@Q}
out_dir = ${OUT_DIR@Q}
height = int(${HEIGHT})
width = int(${WIDTH})
frames = int(${FRAMES})
world_size = int(${WORLD_SIZE})

print(f"[hunyuan-vae] model_dir = {model_dir}")
print(f"[hunyuan-vae] out_dir   = {out_dir}")
print(f"[hunyuan-vae] shape     = {(height, width, frames)}")
print(f"[hunyuan-vae] world_size = {world_size}")
print("[hunyuan-vae] tp_degree = 1")

app = NeuronHunyuanVideoApplication(
    model_path=model_dir,
    parallel=DiffletParallelConfig(tp_degree=world_size, cp_degree=1),
    dtype=torch.bfloat16,
    shape={"height": height, "width": width, "num_frames": frames},
    enable_transformer=False,
    enable_vae_decoder=True,
)

t0 = time.time()
app.compile(out_dir, debug=False)
print(f"[hunyuan-vae] compile elapsed = {time.time() - t0:.3f}s")

t1 = time.time()
app.load(out_dir, skip_warmup=True)
print(f"[hunyuan-vae] load elapsed = {time.time() - t1:.3f}s")

vae = app.vae_decoder
assert vae is not None
cfg = vae.config
tile = torch.randn(
    [
        1,
        int(cfg.latent_channels),
        int(cfg.tile_latent_frames),
        int(cfg.tile_latent_height),
        int(cfg.tile_latent_width),
    ],
    dtype=torch.bfloat16,
)
t2 = time.time()
out = vae.forward(tile)
print(f"[hunyuan-vae] tile forward elapsed = {time.time() - t2:.3f}s")
print(f"[hunyuan-vae] tile output shape = {tuple(out.shape)} dtype = {out.dtype}")
print(f"[hunyuan-vae] finite all = {bool(torch.isfinite(out).all())}")
print("[hunyuan-vae] DONE")
PY
