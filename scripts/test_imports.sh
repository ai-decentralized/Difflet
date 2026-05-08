#!/usr/bin/env bash
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

exec "${PYTHON_BIN}" - <<'PY'
import importlib

modules = [
    "nova",
    "nova.registry",
    "nova.pipeline.nova_pipeline",
    "nova.pipeline.compile_cache",
    "nova.pipeline.parallel_config",
    "nova.models.flux.application",
]

for name in modules:
    importlib.import_module(name)
    print(f"ok import {name}")
PY
