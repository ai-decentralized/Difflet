#!/usr/bin/env bash
# Report the CPU reference job: load time, every 10th step, each result, its end or death.
L=/home/ubuntu/campaigns/wan-dual-2026-10-31/logs
reported=""
emit() { echo "$reported" | grep -qF "|$1|" || { echo "$1"; reported="$reported|$1|"; }; }
while true; do
  for name in ref_dual ref_single; do
    [ -f "$L/$name.result" ] && emit "$(cat "$L/$name.result")"
    [ -f "$L/$name.log" ] || continue
    load=$(grep -oE "^load [0-9]+ s.*" "$L/$name.log" | tail -1)
    [ -n "$load" ] && emit "$name $load"
    step=$(tail -c 4000 "$L/$name.log" | tr '\r' '\n' | grep -oE " (10|20|30|40)/40 \[[^]]*\]" | tail -1)
    [ -n "$step" ] && emit "$name step$step"
    grep -q "Traceback" "$L/$name.log" && emit "$name TRACEBACK: $(tail -2 "$L/$name.log" | tr '\n' ' ')"
  done
  [ -f "$L/cpu_ref.done" ] && { cat "$L/cpu_ref.done"; exit 0; }
  kill -0 "$(cat "$L/job_cpu_ref.pid")" 2>/dev/null || { echo "CPU REF JOB DEAD without cpu_ref.done"; exit 1; }
  sleep 60
done
