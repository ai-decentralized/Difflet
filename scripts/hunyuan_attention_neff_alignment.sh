#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python}"

export PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-1}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"
export DIFFLET_RUN_HUNYUAN_ATTENTION_NEFF=1

cd "${ROOT}"
exec "${PYTHON_BIN}" -m pytest tests/numerical/test_hunyuan_video_attention_neff.py -q "$@"
