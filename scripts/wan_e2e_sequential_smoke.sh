#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATENTS_PATH="${DIFFLET_WAN_E2E_LATENTS_PATH:-${ROOT}/.difflet-cache/wan_e2e_latents.pt}"

cd "${ROOT}"

echo "[wan-e2e-seq] stage 1/2: transformer -> latent"
DIFFLET_WAN_E2E_ENABLE_TEXT="${DIFFLET_WAN_E2E_ENABLE_TEXT:-0}" \
DIFFLET_WAN_E2E_ENABLE_VAE=0 \
DIFFLET_WAN_E2E_OUTPUT_TYPE=latent \
DIFFLET_WAN_E2E_SAVE_LATENTS="${LATENTS_PATH}" \
"${ROOT}/scripts/wan_e2e_smoke.sh" "$@"

echo "[wan-e2e-seq] stage 2/2: VAE decode"
NEURON_RT_NUM_CORES="${DIFFLET_WAN_E2E_VAE_NUM_CORES:-1}" \
DIFFLET_WAN_TP_DEGREE=1 \
DIFFLET_WAN_E2E_ENABLE_TEXT=0 \
DIFFLET_WAN_E2E_ENABLE_TRANSFORMER=0 \
DIFFLET_WAN_E2E_ENABLE_TRANSFORMER_2=0 \
DIFFLET_WAN_E2E_ENABLE_VAE=1 \
DIFFLET_WAN_E2E_OUTPUT_TYPE=pt \
DIFFLET_WAN_E2E_LOAD_LATENTS="${LATENTS_PATH}" \
"${ROOT}/scripts/wan_e2e_smoke.sh" "$@"
