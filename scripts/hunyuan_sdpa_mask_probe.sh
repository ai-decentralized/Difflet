#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python}"
export PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:${PATH}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-1}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

exec "${PYTHON_BIN}" scripts/hunyuan_sdpa_mask_probe.py "$@"
