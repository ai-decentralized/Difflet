#!/usr/bin/env bash
# launch.sh <name> <job-script.sh> [args...] : detached job with log / pid / done files under the job tmp dir.
set -uo pipefail
NAME="${1:?}"; JOB="${2:?}"; shift 2
LOGDIR=${PTQ_JOBS:-/home/ubuntu/ptq-jobs}/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/$NAME.log"; PIDFILE="$LOGDIR/$NAME.pid"; DONEFILE="$LOGDIR/$NAME.done"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "launch: $NAME already running as PID $(cat "$PIDFILE")"; exit 1
fi
rm -f "$DONEFILE" "$DONEFILE.seen"; : > "$LOG"
setsid bash -c 'job="$1"; log="$2"; done="$3"; shift 3; bash "$job" "$@" >> "$log" 2>&1; echo $? > "$done"' _ "$JOB" "$LOG" "$DONEFILE" "$@" &
echo $! > "$PIDFILE"
echo "launch: $NAME started PID $(cat "$PIDFILE") log=$LOG"
