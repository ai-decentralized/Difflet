#!/usr/bin/env bash
# Qwen-Image AoT before / after on the NATIVE run's initial noise (the noise ref_native_noise_50
# was made with), so the AoT result is scored on the same ruler as the native fix (32.9 dB).
# Reuses the compiled roots; quality runs "qn" (bench4m dumps initial/final latents for q* runs).
set -u
# Host paths are settings (2026-11-03, moved into the repo): BENCH4M_DIR is the data root (plus
# bench4m.py's own BENCH4M_* settings), PR134_AFTER_WT / PR134_BEFORE_WT the two checkouts,
# NATIVE_NOISE the native run's initial.pt. bench4m.py takes the AoT worktree as BENCH4M_AOT_WT.
D=${BENCH4M_DIR:?set BENCH4M_DIR (see benchmark/bench4m/README.md)}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=${BENCH4M_AOT_VENV:?set BENCH4M_AOT_VENV}/bin:$PATH
NOISE=${NATIVE_NOISE:?set NATIVE_NOISE to the native q1 run initial.pt}
mkdir -p $D/status
for cell_wt in "after_tp4:${PR134_AFTER_WT:?set PR134_AFTER_WT}" "before_tp4:${PR134_BEFORE_WT:?set PR134_BEFORE_WT}"; do
  cell=${cell_wt%%:*}; wt=${cell_wt#*:}
  if [ -f "$D/qwen_image/$cell/qn/summary.json" ]; then continue; fi
  [ -e "$D/qwen_image/$cell/qn" ] && mv "$D/qwen_image/$cell/qn" "$D/qwen_image/$cell/qn.partial-$(date +%s)"
  BENCH4M_AOT_WT=$wt /usr/bin/python3 $HERE/bench4m.py run --model qwen_image --cell "$cell" --run qn \
    --mode generate --venv aot --env DV_INJECT_INITIAL=$NOISE -- --tp-degree 4 --cp-degree 1
  echo "DONE qwen_image/$cell/qn rc=$?" > $D/status/qwen_image__${cell}__qn.txt
done
echo "QN DONE" > $D/qn.done
