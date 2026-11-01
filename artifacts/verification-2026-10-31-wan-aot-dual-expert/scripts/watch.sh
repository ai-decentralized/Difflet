#!/usr/bin/env bash
# Report each new cell result once, the done marker, or the job dying without it. Owns nothing.
C=/home/ubuntu/campaigns/wan-dual-2026-10-31
SEEN=$C/logs/.watch_seen
touch "$SEEN"
while true; do
  for f in "$C"/logs/*.result; do
    [ -f "$f" ] || continue
    grep -qxF "$f" "$SEEN" || { cat "$f"; echo "$f" >> "$SEEN"; }
  done
  if [ -f "$C/logs/cells.done" ]; then cat "$C/logs/cells.done"; exit 0; fi
  if ! kill -0 "$(cat "$C/logs/job_cells.pid" 2>/dev/null)" 2>/dev/null; then
    echo "JOB DEAD without cells.done (see $C/logs/job_cells.out)"; exit 1
  fi
  sleep 30
done
