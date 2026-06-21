#!/usr/bin/env bash
# N4 — HunyuanVideo backbone compile-and-load smoke.
#
# Default config: 4 dual + 8 single + 1 refiner layers at production heads
# (24 x 128), default M3 shape (320x512x61), tp=4, bf16, skip_warmup=False.
#
# Scale to production via env vars:
#   DIFFLET_HUNYUAN_N4_NUM_LAYERS=20
#   DIFFLET_HUNYUAN_N4_NUM_SINGLE_LAYERS=40
#   DIFFLET_HUNYUAN_N4_NUM_REFINER_LAYERS=2
#   DIFFLET_HUNYUAN_N4_METRICS=/tmp/hunyuan_n4_prod.json
#
# Skip the warmup forward (faster iteration, no first-execute coverage):
#   DIFFLET_HUNYUAN_N4_SKIP_WARMUP=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"

if [[ -d "${NEURON_VENV}/bin" ]]; then
  export PATH="${NEURON_VENV}/bin:${PATH}"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-4}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

cd "${ROOT}"
exec "${PYTHON_BIN}" scripts/hunyuan_backbone_compile_load_smoke.py "$@"
