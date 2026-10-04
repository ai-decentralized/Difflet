#!/usr/bin/env bash
# Online-delta alpha sweep job (2026-09-18): for each sweep label (ascending
# alpha), run the campaign cell on every model via drive_parallel.sh, which is
# serial on the device and resumable per metric (cell_status). No compile:
# the alpha is a generate-only flag on the warm tp4 artifact. Meant to run
# under .claude/skills/difflet-device-verify/scripts/supervise.sh, which
# restarts it when a swept task takes it down before the done marker.
#
#   benchmark/trn2/run_tcod_sweep.sh <done-marker> [label ...]
#
# Env: DIFFLET_VENV (the Neuron venv), DIFFLET_BENCH_DEVICE (default trn2),
#      DIFFLET_SWEEP_MODELS (space-separated model slugs; default = all five).
set -u
MARKER="${1:?usage: run_tcod_sweep.sh <done-marker> [label ...]}"
shift
LABELS=("$@")
if [[ ${#LABELS[@]} -eq 0 ]]; then
  LABELS=(tp4tcod02 tp4tcod03 tp4tcod04 tp4tcod05 tp4tcod06 tp4tcod08)
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export DIFFLET_VENV="${DIFFLET_VENV:-$ROOT/.venv}"
export PATH="$DIFFLET_VENV/bin:$PATH"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE="${DIFFLET_BENCH_DEVICE:-trn2}"
# shellcheck disable=SC2206
MODELS=(${DIFFLET_SWEEP_MODELS:-})
ts() { date -u +%FT%TZ; }
fail=0
# preflight: the PSNR reference outputs (tp4 cell, same host) must be present
for f in flux_1_dev_out.png qwen_image_out.png ltx_2_out.mp4 hunyuanvideo_out.mp4 \
         wan2_1_t2v_14b_diffusers_out.mp4; do
  [[ -f "benchmark/$DIFFLET_BENCH_DEVICE/logs/$f" ]] || \
    echo "[sweep] $(ts) WARNING reference output missing: logs/$f (PSNR column will be blank)"
done
for label in "${LABELS[@]}"; do
  echo "[sweep] $(ts) === $label ==="
  if bash benchmark/trn2/drive_parallel.sh "$label" "${MODELS[@]}"; then
    echo "[sweep] $(ts) $label LABEL_DONE"
  else
    echo "[sweep] $(ts) $label LABEL_FAILED (a cell is incomplete; rerun the label to resume)"
    fail=1
  fi
done
echo "[sweep] $(ts) ALL_DONE fail=$fail" | tee "$MARKER"
exit "$fail"
