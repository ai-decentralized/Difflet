#!/usr/bin/env bash
# Sweep-resilient supervisor. Run INSIDE a Monitor tool call:
#   bash supervise.sh <job-script> <done-marker> <name> [report-file...]
# Restarts the job when no process matches "<basename of job-script>" and the
# marker is absent (a swept task); a job that fails writes its marker and is
# reported, never retried. Extra args are per-stage marker files to echo once.
# The pgrep pattern is path-anchored ("/<basename>") so this supervisor's own
# name can never match it.
JOB=$1; MARKER=$2; NAME=$3; shift 3
STAGES=("$@")
rm -f "$MARKER"
seen=""
while [ ! -f "$MARKER" ]; do
  if ! pgrep -f "/$(basename "$JOB")" > /dev/null; then
    bash "$JOB" >> "/tmp/logs/$NAME.supervised.log" 2>&1 &
    echo "$NAME (re)started, pid $!"
    sleep 15
  fi
  for f in "${STAGES[@]}"; do
    if [ -f "$f" ] && ! echo "$seen" | grep -q "$f"; then cat "$f"; seen="$seen $f"; fi
  done
  free_gb=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "${free_gb:-999}" -lt 80 ] && [ ! -f "/tmp/logs/.disk_alerted_$NAME" ]; then
    echo "DISK LOW: ${free_gb}G free on / — stop and ask before deleting any cache"
    touch "/tmp/logs/.disk_alerted_$NAME"
  fi
  sleep 30
done
cat "$MARKER"
