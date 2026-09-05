#!/usr/bin/env bash
# Launch a long job detached, with a PID file and an exit-code file.
#
# Usage: scripts/supervise.sh <name> <job-script.sh>
#
# Writes:
#   /tmp/logs/<name>.log    combined stdout+stderr of the job
#   /tmp/logs/<name>.pid    PID of the detached job leader
#   /tmp/logs/<name>.done   the job's exit code, written when it finishes
#
# Why a PID file rather than pgrep: the bundled supervisor matched its job by
# pgrep on the command line, which also matched the supervisor's own argv, so
# it concluded the job was "already running" and never launched it (fixed on
# the campaign branch as 5fabe13). A PID file cannot alias like that, and
# setsid detaches the job from this shell's process group so it survives the
# caller exiting.
set -uo pipefail

NAME="${1:?usage: supervise.sh <name> <job-script.sh>}"
JOB="${2:?usage: supervise.sh <name> <job-script.sh>}"
LOGDIR=/tmp/logs
mkdir -p "$LOGDIR"

LOG="$LOGDIR/${NAME}.log"
PIDFILE="$LOGDIR/${NAME}.pid"
DONEFILE="$LOGDIR/${NAME}.done"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "supervise: $NAME already running as PID $(cat "$PIDFILE")" >&2
  exit 1
fi

rm -f "$DONEFILE"
: > "$LOG"

setsid bash -c '
  job="$1"; log="$2"; done="$3"
  bash "$job" >> "$log" 2>&1
  echo $? > "$done"
' _ "$JOB" "$LOG" "$DONEFILE" &

echo $! > "$PIDFILE"
echo "supervise: $NAME started (PID $(cat "$PIDFILE")), log=$LOG done=$DONEFILE"
