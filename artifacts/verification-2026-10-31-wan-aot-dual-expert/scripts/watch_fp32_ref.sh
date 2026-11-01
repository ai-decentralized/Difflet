#!/usr/bin/env bash
# Report the fp32 reference job: each result, expert loads, every 10th fp32 step, end or death.
L=/home/ubuntu/campaigns/wan-dual-2026-10-31/logs
reported=""
emit() { echo "$reported" | grep -qF "|$1|" || { echo "$1"; reported="$reported|$1|"; }; }
while true; do
  for name in check_swap ref_dual_fp32; do
    [ -f "$L/$name.result" ] && emit "$(cat "$L/$name.result")"
    [ -f "$L/$name.log" ] || continue
    while read -r line; do emit "$name $line"; done < <(grep -E "^step [0-9]+: loaded|^step (10|20|30|40)/40|Traceback" "$L/$name.log")
  done
  [ -f "$L/fp32_ref.done" ] && { cat "$L/fp32_ref.done"; exit 0; }
  kill -0 "$(cat "$L/job_fp32_ref.pid")" 2>/dev/null || { echo "FP32 REF JOB DEAD without fp32_ref.done"; exit 1; }
  sleep 60
done
