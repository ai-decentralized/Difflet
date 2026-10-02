#!/usr/bin/env bash
# watch_verify.sh <name> : emit the runner's step markers from logs/<name>.log until logs/<name>.done appears.
N=${1:?}
L=${PTQ_JOBS:-/home/ubuntu/ptq-jobs}/logs/$N.log
D=${PTQ_JOBS:-/home/ubuntu/ptq-jobs}/logs/$N.done
PAT='^=== |_RC=|VERIFY_DONE|host busy|Traceback|Error:|error:|status=failed|\[bench\]|\[cold\]|\[warm\]|\[done\]|psnr'
seen=0
while true; do
  n=$(grep -cE "$PAT" "$L" 2>/dev/null)
  if [ "${n:-0}" -gt "$seen" ]; then grep -E "$PAT" "$L" | sed -n "$((seen+1)),${n}p" | cut -c1-170; seen=$n; fi
  if [ -f "$D" ]; then echo "$N finished rc=$(cat "$D")"; exit 0; fi
  sleep 30
done
