#!/usr/bin/env bash
# M2.5-A Wan numerical alignment gate.
#
# This is intentionally component-scoped. Full HF Wan2.2 T2V reference loads
# UMT5 plus large transformer weights and is too memory-sensitive for the
# regular 4-core spike box. M2.5-A locks the pieces we can verify repeatably:
# pure-torch Wan reference parity tests and VAE real-weight CPU/NEFF alignment.
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

cd "${ROOT}"

echo "[wan-m25] reference parity unit checks"
"${PYTHON_BIN}" -m pytest -q \
  tests/unit/test_modeling_wan.py::test_wan_rotary_pos_embed_shapes_match_diffusers_reference \
  tests/unit/test_wan_pipeline_orchestrator.py::test_negative_prompt_string_routes_through_tokenizer

echo "[wan-m25] VAE real-weight CPU + NEFF numerical alignment"
DIFFLET_WAN_VAE_RUN_NEFF_NUMERIC="${DIFFLET_WAN_VAE_RUN_NEFF_NUMERIC:-1}" \
  ./scripts/wan_vae_real_alignment.sh

echo "[wan-m25] PASS: M2.5-A component numerical alignment complete"
