# Lane helpers for scripts/profile_dit (sourced). A lane is one device (logical cores 4d..4d+3)
# and one directory <root>/<key>/<arm>/ whose STATUS file ends it: DONE, FAILED <reason> or BUSY.
# Every process a lane starts on the device carries DIFFLET_PROF_LANE=<that directory>, so a
# restarted lane finds its own leftovers by environment, never by a pid that may have been reused.

WORLD=4  # capture workers = world size = tp x cfg x cp, every key
# One `neuron-explorer capture` at a time among our lanes, host-wide (ruling R17): a capture is a
# measurement, so no other lane's replay runs beside it; dumps and views stay parallel.
CAPTURE_LOCK=${CAPTURE_LOCK:-/mnt/nvme/difflet-prof/.capture.lock}

note() {  # note <lane dir> <text...>: a timestamped line in the lane log
  local dir=$1; shift
  echo "$(date '+%F %T') $*" >> "$dir/lane.log"
}

write_status() {  # write_status <lane dir> <DONE | FAILED <reason> | BUSY>
  local dir=$1; shift
  echo "$*" > "$dir/STATUS"
  note "$dir" "STATUS $*"
}

require_idle() {  # require_idle <device> <lane dir>: kills the lane's leftover processes, then
                  # returns 0 if no process holds /dev/neuron<device>; else writes BUSY, returns 1
  local d=$1 dir=$2 left _
  left=$(grep -lszxF "DIFFLET_PROF_LANE=$dir" /proc/[0-9]*/environ | cut -d/ -f3)
  if [ -n "$left" ]; then
    note "$dir" "killing the lane's leftover processes" $left
    kill -KILL $left 2>/dev/null
    for _ in $(seq 60); do kill -0 $left 2>/dev/null || break; sleep 1; done
  fi
  note "$dir" "fuser /dev/neuron$d:"
  if fuser "/dev/neuron$d" >> "$dir/lane.log" 2>&1; then  # exit 0: some process holds it
    write_status "$dir" BUSY
    return 1
  fi
  note "$dir" "/dev/neuron$d idle"
}

cores_busy() {  # cores_busy <log>: the runtime refused the cores (failed_laneB's line)
  grep -qF 'Logical Neuron Core(s) not available' "$1"
}

capture_neff() {  # capture_neff <device> <lane dir> <blk dir> <cpus>: <blk dir>/block.neff replayed
                  # with its collectives as WORLD workers on the device's cores, executions 4 and 8
                  # profiled (<blk dir>/profile_rank_0_exec_{4,8}.ntff), holding CAPTURE_LOCK from
                  # require_idle to the capture's end. Returns 1 when the device is busy (BUSY
                  # written), else 0 with CAPTURE_RC = the capture's exit (capture.log's last line)
  local d=$1 dir=$2 blk=$3 cpus=$4
  exec 8>> "$CAPTURE_LOCK"
  flock 8
  note "$dir" "capture lock taken"
  if ! require_idle "$d" "$dir"; then flock -u 8; exec 8>&-; return 1; fi
  # --enable-dge-notifs: the dynamic DMAs traced (without it the profile warns its DMA values
  # may be inaccurate); same total_time, no slower (README, Task 1 smokes)
  ( cd "$blk" && env DIFFLET_PROF_LANE="$dir" NEURON_RT_VISIBLE_CORES=$((4 * d))-$((4 * d + 3)) \
      taskset -c "$cpus" timeout -k 60 1200 neuron-explorer capture -n block.neff \
      -s "$blk/profile.ntff" --collectives-worker-count=$WORLD \
      --collectives-workers-per-node=$WORLD --collectives-worker-start-id=0 \
      --collectives-profile-id=0 --num-exec=8 --profile-nth-exec=4 --enable-dge-notifs \
  ) > "$blk/capture.log" 2>&1
  CAPTURE_RC=$?
  flock -u 8  # a leftover child holding fd 8 must not keep the lock
  exec 8>&-
  echo "capture exit $CAPTURE_RC" >> "$blk/capture.log"
}
