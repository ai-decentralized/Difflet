#!/usr/bin/env bash
# Unit suite. Prefers the Neuron venv at /opt when present; otherwise `python`
# on PATH. Hosts without the Neuron SDK (laptops, the CUDA box, CI): set
# PYTHON_BIN=<repo>/.venv/bin/python and DIFFLET_BACKEND=cpu — auto-detect falls
# back to trainium on a host with no accelerator. Tests that need the SDK skip
# there with a reason (see tests/conftest.py).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="python"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
NEURON_PYTHON="${NEURON_VENV}/bin/python"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${NEURON_PYTHON}" ]]; then
    PYTHON_BIN="${NEURON_PYTHON}"
  else
    PYTHON_BIN="${DEFAULT_PYTHON}"
  fi
fi

if [[ -d "${NEURON_VENV}/bin" ]]; then
  export PATH="${NEURON_VENV}/bin:${PATH}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

exec "${PYTHON_BIN}" -m pytest tests/unit -q "$@"
