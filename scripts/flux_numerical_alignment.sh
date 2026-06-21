#!/usr/bin/env bash
# Opt-in Flux trajectory numerical alignment against Hugging Face diffusers.
#
# Defaults use the existing 1024x1024 Flux compile shape with a short 4-step
# trajectory. Override DIFFLET_FLUX_NUMERICAL_STEPS=28 for the full release gate.
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
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-4}"
export DIFFLET_RUN_FLUX_NUMERICAL=1
export DIFFLET_FLUX_NUMERICAL_MODEL="${DIFFLET_FLUX_NUMERICAL_MODEL:-black-forest-labs/FLUX.1-dev}"
export DIFFLET_FLUX_NUMERICAL_HEIGHT="${DIFFLET_FLUX_NUMERICAL_HEIGHT:-1024}"
export DIFFLET_FLUX_NUMERICAL_WIDTH="${DIFFLET_FLUX_NUMERICAL_WIDTH:-1024}"
export DIFFLET_FLUX_NUMERICAL_STEPS="${DIFFLET_FLUX_NUMERICAL_STEPS:-4}"
export DIFFLET_FLUX_NUMERICAL_TP_DEGREE="${DIFFLET_FLUX_NUMERICAL_TP_DEGREE:-4}"
export DIFFLET_FLUX_NUMERICAL_MIN_COSINE="${DIFFLET_FLUX_NUMERICAL_MIN_COSINE:-0.95}"
export DIFFLET_FLUX_NUMERICAL_METRICS="${DIFFLET_FLUX_NUMERICAL_METRICS:-/tmp/difflet_flux_numerical_metrics.json}"

cd "${ROOT}"
exec "${PYTHON_BIN}" -m pytest -q tests/numerical/test_flux_vs_diffusers.py "$@"
