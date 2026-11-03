#!/usr/bin/env bash
# PR #134 on-device verification (2026-11-01): AoT (trainium) TP4, before vs after, Qwen-Image and
# HunyuanVideo 1.0. Sequential, one device user at a time (bench4m waits for /dev/neuron0 idle).
# Re-runnable under supervise.sh: a run with rc 0 is skipped; a run with rc != 0 stops the job
# (genuine failure, reported, never retried); a run dir without summary.json (killed mid-run) is
# moved aside, and for a compile its root too, then the run starts over.
set -u
# Host paths are settings (2026-11-03, moved into the repo): BENCH4M_DIR is the data root (plus
# bench4m.py's own BENCH4M_* settings), PR134_AFTER_WT / PR134_BEFORE_WT the two checkouts.
# bench4m.py now takes the AoT worktree as BENCH4M_AOT_WT (it was DV_AOT_WT).
D=${BENCH4M_DIR:?set BENCH4M_DIR (see benchmark/bench4m/README.md)}
H="/usr/bin/python3 $(cd "$(dirname "$0")" && pwd)/bench4m.py"
export PATH=${BENCH4M_AOT_VENV:?set BENCH4M_AOT_VENV}/bin:$PATH
AFTER=${PR134_AFTER_WT:?set PR134_AFTER_WT}
BEFORE=${PR134_BEFORE_WT:?set PR134_BEFORE_WT}
MARKER=$D/job.done
mkdir -p $D/status

finish() { echo "$1" > "$MARKER"; exit 0; }

# step <model> <cell> <wt> <run> <mode> <flags...>
step() {
  local model=$1 cell=$2 wt=$3 run=$4 mode=$5; shift 5
  local dir=$D/$model/$cell/$run root=$D/roots/${model}_${cell} ts
  local status=$D/status/${model}__${cell}__${run}.txt
  if [ -f "$dir/summary.json" ]; then
    rc=$(/usr/bin/python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['rc'])" "$dir/summary.json")
    if [ "$rc" = 0 ]; then return 0; fi
    finish "FAIL $model/$cell/$run rc=$rc (see $dir/run.log)"
  fi
  ts=$(date +%s)
  if [ -e "$dir" ]; then mv "$dir" "$dir.partial-$ts"; fi
  if [ "$mode" = compile ] && [ -e "$root" ]; then mv "$root" "$root.partial-$ts"; fi
  BENCH4M_AOT_WT=$wt $H run --model "$model" --cell "$cell" --run "$run" --mode "$mode" --venv aot -- "$@"
  rc=$?
  wall=$(/usr/bin/python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('wall_s'))" "$dir/summary.json" 2>/dev/null)
  echo "DONE $model/$cell/$run rc=$rc wall_s=$wall" > "$status"
  if [ "$rc" != 0 ]; then finish "FAIL $model/$cell/$run rc=$rc (see $dir/run.log)"; fi
}

QC="--backend trainium --tp-degree 4 --cp-degree 1"
QG="--tp-degree 4 --cp-degree 1"
HVF="--tp-degree 4 --cp-degree 1"

# Qwen-Image first: the masked graph compiling is the main risk, so "after" goes first.
for cell_wt in "after_tp4:$AFTER" "before_tp4:$BEFORE"; do
  cell=${cell_wt%%:*}; wt=${cell_wt#*:}
  step qwen_image "$cell" "$wt" compile compile $QC
  for run in gen0 gen1 gen2 q1; do step qwen_image "$cell" "$wt" "$run" generate $QG; done
done
for cell_wt in "after_tp4:$AFTER" "before_tp4:$BEFORE"; do
  cell=${cell_wt%%:*}; wt=${cell_wt#*:}
  step hunyuan_video "$cell" "$wt" compile compile $HVF
  for run in gen0 gen1 q1; do step hunyuan_video "$cell" "$wt" "$run" generate $HVF; done
done
finish "ALL DONE"
