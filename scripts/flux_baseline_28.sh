#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"

export PATH="${NEURON_VENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export NEURON_RT_LOG_LEVEL="${NEURON_RT_LOG_LEVEL:-ERROR}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-4}"

cd "${ROOT}"
exec "${PYTHON_BIN}" examples/flux_example.py \
  --model "${MODEL_ID:-black-forest-labs/FLUX.1-dev}" \
  --tp-degree "${TP_DEGREE:-4}" \
  --skip-warmup \
  --num-inference-steps "${NUM_INFERENCE_STEPS:-28}" \
  --prompt "${PROMPT:-a cat}" \
  --output "${OUTPUT:-/tmp/flux_28step_baseline.png}" \
  "$@"
