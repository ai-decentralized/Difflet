#!/usr/bin/env bash
# M3 v0 HunyuanVideo end-to-end smoke wrapper.
#
# Drives the hybrid path in one process:
#
#   cached DiT input artifact -> Difflet Trainium DiT (4 steps)
#                             -> HF AutoencoderKLHunyuanVideo decode (CPU)
#                             -> (1, 3, T, H, W) bf16 video tensor
#                             -> best-effort MP4 (cclog 29b §4)
#
# Prerequisites:
#
# 1. Compiled HunyuanVideo backbone (cclog 28/29 §7.3):
#       .difflet-cache/hunyuan_n4_20d40s2r/compiled/transformer/{model.pt,neuron_config.json}
# 2. Real HF HunyuanVideo source (cclog 29 §7.4 / 29b §4):
#       ~/.cache/huggingface/hub/hunyuanvideo-real/transformer/
#       ~/.cache/huggingface/hub/hunyuanvideo-real/vae/
# 3. Cached DiT input artifact (cclog 29 §7.3):
#       .difflet-cache/hunyuan_dit_inputs/<name>.safetensors + .meta.json
#       generate via `scripts/hunyuan_video_cache_dit_inputs.py`
#
# All four paths overridable via env vars below.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"

if [[ -d "${NEURON_VENV}/bin" ]]; then
  export PATH="${NEURON_VENV}/bin:${PATH}"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-4}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

SOURCE_DIR="${DIFFLET_HUNYUAN_SOURCE_DIR:-/home/ubuntu/.cache/huggingface/hub/hunyuanvideo-real}"
COMPILED_DIR="${DIFFLET_HUNYUAN_COMPILED_DIR:-${ROOT}/.difflet-cache/hunyuan_n4_20d40s2r/compiled}"
BUNDLE="${DIFFLET_HUNYUAN_BUNDLE:-${ROOT}/.difflet-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors}"
OUTPUT="${DIFFLET_HUNYUAN_OUTPUT:-/tmp/hunyuan_smoke.mp4}"

cd "${ROOT}"
exec "${PYTHON_BIN}" scripts/hunyuan_smoke.py \
  --source-dir "${SOURCE_DIR}" \
  --compiled-dir "${COMPILED_DIR}" \
  --bundle "${BUNDLE}" \
  --output "${OUTPUT}" \
  "$@"
