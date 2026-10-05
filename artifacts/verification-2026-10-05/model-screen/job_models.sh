#!/usr/bin/env bash
# Levers on the other models, 2026-10-05 (trn2.3xlarge, tp4), through the production CLI.
#   LTX-2: bf16 base vs bf16 + VC2 + strided DMA (VC is not in the LTX-2 cache key -> own cache
#          dir), then fp8 static with the levers.
#   HunyuanVideo / Qwen-Image / FLUX (VC2 already on): fp8 static with and without strided DMA
#          (tensorizer extras are in their cache keys, so one cache dir; only the DiT recompiles).
# Serialised after the Wan defaults A/B. Results: <model-screen>/<name>/result.json.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PATH="/home/ubuntu/Difflet/.venv/bin:${PATH}"
export PYTHONPATH="${ROOT}"
S="${ROOT}/scripts/dit_step_screen.sh"
export OUT_DIR="${ROOT}/artifacts/verification-2026-10-05/model-screen"
CAL="${ROOT}/artifacts/verification-2026-10-03/fp8-dyn-step/static"
HVCAL="${ROOT}/artifacts/verification-2026-10-03/hv-fp8/static/act_calibration_hv.json"
exec >> "${OUT_DIR}/job_models.out" 2>&1
while pgrep -f 'job_defaults.sh|ptq_fp8_ab.py' >/dev/null; do sleep 30; done
run() { echo "=== $(date -u +%T) $1"; "$S" "$@" || echo "!!! $1 failed"; }

# --- LTX-2 (480x704x49, 20 steps, guidance 1.0; device VAE, no host-vae path)
LTX=(--model-id Lightricks/LTX-2 --revision dfcc2108383fe1aaa0584bdf55d368a4bdadd90c --tp-degree 4 --height 480 --width 704 --num-frames 49)
export OUT_EXT=mp4 STEPS=20 GUIDANCE=1.0
run ltx2_bf16_base    -- "${LTX[@]}"
run ltx2_bf16_levers  NEURON_RT_VIRTUAL_CORE_SIZE=2 DIFFLET_STRIDED_DMA=1 -- "${LTX[@]}" --cache-dir /home/ubuntu/.cache/difflet-ltx2-levers
run ltx2_fp8_levers   NEURON_RT_VIRTUAL_CORE_SIZE=2 DIFFLET_STRIDED_DMA=1 -- "${LTX[@]}" --cache-dir /home/ubuntu/.cache/difflet-ltx2-levers --quant fp8 --quant-calibration "${CAL}/act_calibration_ltx_2.json"

# --- HunyuanVideo (320x512x61, 20 steps, guidance 6.0, host VAE)
HV=(--model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773 --tp-degree 4 --height 320 --width 512 --num-frames 61 --host-vae --quant fp8 --quant-calibration "${HVCAL}")
export OUT_EXT=mp4 STEPS=20 GUIDANCE=6.0
run hv_fp8_base  -- "${HV[@]}"
run hv_fp8_sdma  DIFFLET_STRIDED_DMA=1 -- "${HV[@]}"

# --- Qwen-Image (1024x1024, 20 steps, guidance 4.0)
QW=(--model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 --tp-degree 4 --height 1024 --width 1024 --quant fp8 --quant-calibration "${CAL}/act_calibration_qwen_image.json")
export OUT_EXT=png STEPS=20 GUIDANCE=4.0
run qwen_fp8_base -- "${QW[@]}"
run qwen_fp8_sdma DIFFLET_STRIDED_DMA=1 -- "${QW[@]}"

# --- FLUX.1-dev (1024x1024, 28 steps, guidance 3.5)
FX=(--model-id black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 --tp-degree 4 --height 1024 --width 1024 --quant fp8 --quant-calibration "${CAL}/act_calibration_flux.json")
export OUT_EXT=png STEPS=28 GUIDANCE=3.5
run flux_fp8_base -- "${FX[@]}"
run flux_fp8_sdma DIFFLET_STRIDED_DMA=1 -- "${FX[@]}"

echo "=== $(date -u +%T) MODELS_DONE"
