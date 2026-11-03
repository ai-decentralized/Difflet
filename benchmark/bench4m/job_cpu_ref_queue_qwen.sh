#!/usr/bin/env bash
# Plan phase 4, stage "Qwen-Image": the two 50-step CPU fp32 references for Qwen-Image only,
# one after the other: (1) AoT noise (aot_main_tp4/q1 initial.pt c2d9d8b3), (2) native noise
# (native_tp4_hashcheck/q1 initial.pt 95a4306b). Same commands as the first two entries of
# job_cpu_ref_queue_qwen_ltx2.sh (which a later stage can resume: it skips runs that have an rc).
#   systemd-run --user --unit=cpuref_qwen_queue --collect -- bash job_cpu_ref_queue_qwen.sh
set -u
# Host paths are settings (2026-11-03, moved into the repo): BENCH4M_DIR is the data root,
# job_cpu_ref.sh sits next to this file, and the snapshot is HF_HOME's (or QWEN_SNAPSHOT).
B=${BENCH4M_DIR:?set BENCH4M_DIR (see benchmark/bench4m/README.md)}
J=$(cd "$(dirname "$0")" && pwd)/job_cpu_ref.sh
QWEN=${QWEN_SNAPSHOT:-${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6}
Q_AOT=$B/qwen_image/aot_main_tp4/q1
Q_NAT=$B/qwen_image/native_tp4_hashcheck/q1
qlog=$B/cpu_ref_queue_qwen.log
run() {  # model name args...
  local model=$1 name=$2; shift 2
  if [ -e "$B/$model/cpu_ref/$name/rc" ]; then
    echo "$(date -u +%FT%TZ) skip $model/$name (rc $(cat "$B/$model/cpu_ref/$name/rc"))" >> "$qlog"; return 0
  fi
  echo "$(date -u +%FT%TZ) start $model/$name" >> "$qlog"
  bash "$J" "$model" "$name" -- "$@"
  local rc=$?
  echo "$(date -u +%FT%TZ) end $model/$name rc=$rc" >> "$qlog"
  return $rc
}
run qwen_image ref_aot_noise_50 --model-path "$QWEN" --initial "$Q_AOT/initial.pt" --steps 50 || exit 1
run qwen_image ref_native_noise_50 --model-path "$QWEN" --initial "$Q_NAT/initial.pt" --steps 50 || exit 1
touch "$B/cpu_ref_queue_qwen.done"
echo "$(date -u +%FT%TZ) queue done" >> "$qlog"
