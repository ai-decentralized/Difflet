#!/usr/bin/env bash
# Start $C/<job>.sh (default job_cells) fully detached (setsid -f: reparented to init), so no
# tool's process tree owns it and a monitor's expiry cannot kill it. Refuses a second copy.
C=/home/ubuntu/campaigns/wan-dual-2026-10-31
JOB=${1:-job_cells}
PIDFILE=$C/logs/$JOB.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: pid $(cat "$PIDFILE")"; exit 0
fi
setsid -f bash -c "echo \$\$ > '$PIDFILE'; exec bash '$C/$JOB.sh'" \
  >> "$C/logs/$JOB.out" 2>&1 < /dev/null
sleep 2
echo "started $JOB: pid $(cat "$PIDFILE")"
