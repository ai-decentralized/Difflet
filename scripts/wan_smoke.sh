#!/usr/bin/env bash
# M2 Wan spike closure smoke (cclogs/09 §7.4).
#
# Drives the W4 command end-to-end on a 4-NeuronCore Trainium instance.
# 4-core trn3pd98.3xlarge cannot host TP=4 transformer + TP=1 VAE in one
# process (cclogs/16 §7.7), so this wrapper splits into two stages:
#
#   stage 1  text + transformer  -> latents.pt   (TP=4, NEURON_RT_NUM_CORES=4)
#   stage 2  vae decode          -> mp4 / pt     (TP=1, NEURON_RT_NUM_CORES=1)
#
# Forward returns a ``(C, T, H, W)`` tensor at the spike shape; saving
# the MP4 is best-effort and cannot fail the script (per 09 §W4 exit:
# "saving as MP4 is allowed to fail; we only care about the tensor").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"

if [[ -d "${NEURON_VENV}/bin" ]]; then
  export PATH="${NEURON_VENV}/bin:${PATH}"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NOVA_BACKEND="${NOVA_BACKEND:-trainium}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

MODEL="${NOVA_WAN_MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
HEIGHT="${NOVA_WAN_HEIGHT:-480}"
WIDTH="${NOVA_WAN_WIDTH:-832}"
FRAMES="${NOVA_WAN_FRAMES:-9}"
TP_DEGREE="${NOVA_WAN_TP_DEGREE:-4}"
NUM_STEPS="${NOVA_WAN_STEPS:-2}"
PROMPT="${NOVA_WAN_PROMPT:-a cat walking}"
LATENTS_PATH="${NOVA_WAN_LATENTS_PATH:-${ROOT}/.nova-cache/wan_smoke_latents.pt}"
OUTPUT="${NOVA_WAN_OUTPUT:-/tmp/wan_smoke.mp4}"
DOWNLOAD_FLAG=""
if [[ "${NOVA_WAN_DOWNLOAD_WEIGHTS:-0}" == "1" ]]; then
  DOWNLOAD_FLAG="--download-weights"
fi

cd "${ROOT}"

echo "[wan-smoke] stage 1/2: text + transformer -> latents"
NEURON_RT_NUM_CORES=${NOVA_WAN_TRANSFORMER_NUM_CORES:-4} \
  "${PYTHON_BIN}" examples/wan_example.py \
    --model "${MODEL}" \
    --tp-degree "${TP_DEGREE}" \
    --skip-warmup \
    --num-inference-steps "${NUM_STEPS}" \
    --num-frames "${FRAMES}" --height "${HEIGHT}" --width "${WIDTH}" \
    --prompt "${PROMPT}" \
    --enable-text --enable-transformer --no-vae \
    --output-type latent \
    --save-latents "${LATENTS_PATH}" \
    --compiled-dir "${ROOT}/.nova-cache/wan_smoke_stage1" \
    ${DOWNLOAD_FLAG}

echo "[wan-smoke] stage 2/2: vae decode -> tensor (mp4 best-effort)"
NEURON_RT_NUM_CORES=${NOVA_WAN_VAE_NUM_CORES:-1} \
  "${PYTHON_BIN}" examples/wan_example.py \
    --model "${MODEL}" \
    --tp-degree 1 \
    --skip-warmup \
    --num-frames "${FRAMES}" --height "${HEIGHT}" --width "${WIDTH}" \
    --no-text --no-transformer --enable-vae \
    --load-latents "${LATENTS_PATH}" \
    --output-type pt \
    --output "${OUTPUT}" \
    --compiled-dir "${ROOT}/.nova-cache/wan_smoke_stage2" \
    ${DOWNLOAD_FLAG}

FALLBACK_OUTPUT="${OUTPUT%.*}.pt"
if [[ -f "${OUTPUT}" ]]; then
  ACTUAL_OUTPUT="${OUTPUT}"
elif [[ -f "${FALLBACK_OUTPUT}" ]]; then
  ACTUAL_OUTPUT="${FALLBACK_OUTPUT} (mp4 export skipped; tensor only)"
else
  echo "[wan-smoke] WARNING: neither ${OUTPUT} nor ${FALLBACK_OUTPUT} exists" >&2
  ACTUAL_OUTPUT="<missing>"
fi
echo "[wan-smoke] success: latents=${LATENTS_PATH} output=${ACTUAL_OUTPUT}"
