#!/usr/bin/env bash
# Sweep-resilient supervisor. Run INSIDE a Monitor tool call:
#   bash supervise.sh <job-script> <done-marker> <name> [report-file...]
# Restarts the job when it is not alive and the marker is absent (a swept task);
# a job that fails writes its marker and is reported, never retried. Extra args
# are per-stage marker files to echo once.
#
# Liveness is tracked by the PID of the job WE launched, recorded in
# /tmp/logs/<name>.pid — never by pgrep on the job's name. A pgrep pattern built
# from the job path always matches this supervisor too, because the supervisor's
# own argv contains that path ("bash supervise.sh /…/job.sh …"), so the check
# would report "already running" forever and the job would never start.
# setsid detaches the job into its own session so it survives if the supervisor
# (or the Monitor task running it) is killed; the PID file lets a restarted
# supervisor adopt a job that is still running.
JOB=$1; MARKER=$2; NAME=$3; shift 3
STAGES=("$@")
PIDFILE="/tmp/logs/$NAME.pid"
mkdir -p /tmp/logs
rm -f "$MARKER"
seen=""

job_alive() {
  local p
  p=$(cat "$PIDFILE" 2>/dev/null) || return 1
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null
}

while [ ! -f "$MARKER" ]; do
  if ! job_alive; then
    setsid bash "$JOB" >> "/tmp/logs/$NAME.supervised.log" 2>&1 &
    echo $! > "$PIDFILE"
    echo "$NAME (re)started, pid $(cat "$PIDFILE")"
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
rm -f "$PIDFILE"
