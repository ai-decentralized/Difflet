# Lane helpers for scripts/profile_dit (sourced). A lane is one device (logical cores 4d..4d+3)
# and one directory <root>/<key>/<arm>/ whose STATUS file ends it: DONE, FAILED <reason> or BUSY.
# Every process a lane starts on the device carries DIFFLET_PROF_LANE=<that directory>, so a
# restarted lane finds its own leftovers by environment, never by a pid that may have been reused.

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
