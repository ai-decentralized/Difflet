#!/usr/bin/env bash
# Qwen-Image AoT before / after on the NATIVE run's initial noise (the noise ref_native_noise_50
# was made with), so the AoT result is scored on the same ruler as the native fix (32.9 dB).
# Reuses the compiled roots; quality runs "qn" (bench4m dumps initial/final latents for q* runs).
set -u
D=/home/ubuntu/dv-2026-11-01-aot
export PATH=/home/ubuntu/venvs/difflet-neuron/bin:$PATH
NOISE=/home/ubuntu/bench-2026-10-26-4m/qwen_image/native_tp4_hashcheck/q1/initial.pt
for cell_wt in "after_tp4:/home/ubuntu/Difflet/.claude/worktrees/aot-followups" "before_tp4:/home/ubuntu/dv-aot-before"; do
  cell=${cell_wt%%:*}; wt=${cell_wt#*:}
  if [ -f "$D/qwen_image/$cell/qn/summary.json" ]; then continue; fi
  [ -e "$D/qwen_image/$cell/qn" ] && mv "$D/qwen_image/$cell/qn" "$D/qwen_image/$cell/qn.partial-$(date +%s)"
  DV_AOT_WT=$wt /usr/bin/python3 $D/scripts/bench4m.py run --model qwen_image --cell "$cell" --run qn \
    --mode generate --venv aot --env DV_INJECT_INITIAL=$NOISE -- --tp-degree 4 --cp-degree 1
  echo "DONE qwen_image/$cell/qn rc=$?" > $D/status/qwen_image__${cell}__qn.txt
done
echo "QN DONE" > $D/qn.done
