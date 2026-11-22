#!/usr/bin/env bash
# The CPU fp32 references at the const-max campaign's long shapes
# (docs/plans/2026-11-20-cpu-fp32-reference-long.md): two workers, one per NUMA node, 96 threads
# each, pulling keys from one queue, longest first (HV1 and Wan 2.1 start together, then Wan 2.2
# and HV1.5). Every process runs at nice 19: the host is shared and the other project's jobs come
# first. Score a device run against a finished reference with scripts/compare_difflet_runs.py
# (--ref-image <key>/out.mp4 --ref-latents <key>/latents.pt). Runs from a git-archive snapshot
# of one commit (src/, src.sha), so edits to the checkout during the ~30 h cannot reach a run.
#
# usage: run_long_queue.sh setup   snapshot this checkout's HEAD to $B/src and write $B/queue.txt
#        run_long_queue.sh <0|1>   one worker: 0 on node 0 (CPUs 0-47,96-143), 1 on node 1
#                                  (48-95,144-191); progress in $B/progress.txt
# env: B (default /mnt/nvme/difflet-exp/fp32_ref_long; never the root disk)
set -u
B=${B:-/mnt/nvme/difflet-exp/fp32_ref_long}
SRC=$B/src
PY=/home/ubuntu/venvs/native-fp8/bin/python
if [ "${1:-}" = setup ]; then
  # the top level: run inside a subdirectory, git archive would take only that subdirectory
  REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel) || exit 1
  if [ -e "$SRC" ]; then echo "$SRC exists; remove it to snapshot again" >&2; exit 2; fi
  mkdir -p "$SRC"
  git -C "$REPO" archive HEAD | tar -x -C "$SRC" || exit 1
  git -C "$REPO" rev-parse HEAD > "$B/src.sha"
  printf '%s\n' hunyuan_video_long wan21_long wan22_long hunyuan_video_15_long > "$B/queue.txt"
  exit 0
fi
case ${1:-} in
  0) CPUS=0-47,96-143 ;;
  1) CPUS=48-95,144-191 ;;
  *) echo "usage: run_long_queue.sh setup | <0|1>" >&2; exit 2 ;;
esac
[ -d "$SRC" ] || { echo "no $SRC: run 'run_long_queue.sh setup' first" >&2; exit 2; }
next() {  # pop the first line of the queue under a lock
  ( flock 9; head -n 1 "$B/queue.txt"; sed -i '1d' "$B/queue.txt" ) 9> "$B/queue.lock"
}
log() { echo "$(date '+%F %T') worker $W $*" >> "$B/progress.txt"; }
W=$1
while :; do
  key=$(next)
  [ -z "$key" ] && break
  out=$B/$key
  mkdir -p "$out"
  { date '+%F %T'; echo "src $(cat "$B/src.sha") worker $W cpus $CPUS"; uptime; free -g; } > "$out/env.txt"
  log "start $key"
  ( cd "$SRC" && nice -n 19 env TORCH_DEVICE_BACKEND_AUTOLOAD=0 DIFFLET_BACKEND=cpu \
      HF_HUB_OFFLINE=1 PYTHONPATH="$SRC" /usr/bin/time -v taskset -c "$CPUS" \
      "$PY" "$SRC/scripts/fp32_ref/$key.py" --out-dir "$out" --threads 96 ) > "$out/run.log" 2>&1
  rc=$?
  log "done $key exit $rc"
done
log finished
