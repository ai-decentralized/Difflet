#!/usr/bin/env bash
# Opt-in Flux trajectory numerical alignment against Hugging Face diffusers.
#
# Defaults use the existing 1024x1024 Flux compile shape with a short 4-step
# trajectory. Override NOVA_FLUX_NUMERICAL_STEPS=28 for the full release gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
if [[ -d "${NEURON_VENV}/bin" ]]; then
  export PATH="${NEURON_VENV}/bin:${PATH}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NOVA_BACKEND="${NOVA_BACKEND:-trainium}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-4}"
export NOVA_RUN_FLUX_NUMERICAL=1
export NOVA_FLUX_NUMERICAL_MODEL="${NOVA_FLUX_NUMERICAL_MODEL:-black-forest-labs/FLUX.1-dev}"
export NOVA_FLUX_NUMERICAL_HEIGHT="${NOVA_FLUX_NUMERICAL_HEIGHT:-1024}"
export NOVA_FLUX_NUMERICAL_WIDTH="${NOVA_FLUX_NUMERICAL_WIDTH:-1024}"
export NOVA_FLUX_NUMERICAL_STEPS="${NOVA_FLUX_NUMERICAL_STEPS:-4}"
export NOVA_FLUX_NUMERICAL_TP_DEGREE="${NOVA_FLUX_NUMERICAL_TP_DEGREE:-4}"
export NOVA_FLUX_NUMERICAL_MIN_COSINE="${NOVA_FLUX_NUMERICAL_MIN_COSINE:-0.95}"
export NOVA_FLUX_NUMERICAL_METRICS="${NOVA_FLUX_NUMERICAL_METRICS:-/tmp/nova_flux_numerical_metrics.json}"

cd "${ROOT}"
exec "${PYTHON_BIN}" -m pytest -q tests/numerical/test_flux_vs_diffusers.py "$@"
