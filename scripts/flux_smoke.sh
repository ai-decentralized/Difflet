#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
  elif [[ -x "${NEURON_VENV}/bin/python" ]]; then
    PYTHON_BIN="${NEURON_VENV}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

VENV_BIN="$(dirname "${PYTHON_BIN}")"
if [[ -d "${VENV_BIN}" ]]; then
  export PATH="${VENV_BIN}:${PATH}"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export NEURON_RT_LOG_LEVEL="${NEURON_RT_LOG_LEVEL:-ERROR}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-4}"

cd "${ROOT}"
exec "${PYTHON_BIN}" examples/flux_example.py \
  --model "${MODEL_ID:-black-forest-labs/FLUX.1-dev}" \
  --tp-degree "${TP_DEGREE:-4}" \
  --skip-warmup \
  --num-inference-steps "${NUM_INFERENCE_STEPS:-1}" \
  --prompt "${PROMPT:-a cat}" \
  --output "${OUTPUT:-/tmp/flux_smoke.png}" \
  "$@"
