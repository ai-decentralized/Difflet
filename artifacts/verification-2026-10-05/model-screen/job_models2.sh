#!/usr/bin/env bash
# Re-run of ltx2_fp8_levers: its first compile (21:28) ran before commit be5dfe3 put
# virtual_core_size into the LTX-2 cache key, so the generate (21:47, new code) looked for a
# different hash. VC2 + strided DMA are now the LTX-2 defaults, so no switches are needed.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PATH="/home/ubuntu/Difflet/.venv/bin:${PATH}"
export PYTHONPATH="${ROOT}"
S="${ROOT}/scripts/dit_step_screen.sh"
export OUT_DIR="${ROOT}/artifacts/verification-2026-10-05/model-screen"
CAL="${ROOT}/artifacts/verification-2026-10-03/fp8-dyn-step/static"
exec >> "${OUT_DIR}/job_models.out" 2>&1
while pgrep -f 'job_models.sh' >/dev/null; do sleep 60; done
run() { echo "=== $(date -u +%T) $1"; "$S" "$@" || echo "!!! $1 failed"; }
LTX=(--model-id Lightricks/LTX-2 --revision dfcc2108383fe1aaa0584bdf55d368a4bdadd90c --tp-degree 4 --height 480 --width 704 --num-frames 49)
export OUT_EXT=mp4 STEPS=20 GUIDANCE=1.0
run ltx2_fp8_defaults -- "${LTX[@]}" --cache-dir /home/ubuntu/.cache/difflet-ltx2-levers --quant fp8 --quant-calibration "${CAL}/act_calibration_ltx_2.json"
python scripts/ptq_compare_outputs.py --reference "${OUT_DIR}/ltx2_bf16_levers/run1.mp4" --test "${OUT_DIR}/ltx2_fp8_defaults/run1.mp4" \
  --out "${OUT_DIR}/compare_ltx2_fp8_vs_bf16_levers.json" 2>&1 | grep -vE 'Warning|warn' | tail -1
echo "=== $(date -u +%T) MODELS2_DONE"
