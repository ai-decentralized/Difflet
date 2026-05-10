#!/usr/bin/env bash
set -euo pipefail

# Convert a HF Wan2.2 snapshot into Nova-loadable safetensors.
#
# Usage:
#   ./scripts/wan_convert_checkpoint.sh \
#       /path/to/Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/<sha> \
#       /path/to/out
#
# Components default to every directory present in the snapshot that has a
# converter registered (transformer, transformer_2, text_encoder).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
NEURON_PYTHON="${NEURON_VENV}/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_PYTHON}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_DIR="${1:?need model_dir}"
OUT_DIR="${2:?need out_dir}"
COMPONENTS="${NOVA_WAN_COMPONENTS:-}"
OVERWRITE="${NOVA_WAN_OVERWRITE:-0}"

cd "${ROOT}"

exec "${PYTHON_BIN}" - <<PY
import logging
import sys

from nova.models.wan.checkpoint import convert_diffusers_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

components = ${COMPONENTS@Q}
components = components.split(",") if components else None
overwrite = ${OVERWRITE@Q} == "1"

paths = convert_diffusers_checkpoint(
    model_dir=${MODEL_DIR@Q},
    out_dir=${OUT_DIR@Q},
    components=components,
    overwrite=overwrite,
)
print("[wan-convert] wrote:")
for name, path in paths.items():
    print(f"  {name}: {path}")
PY
