#!/usr/bin/env bash
# E2 timing lane: probe.py's arms one after another on one device and one CPU set.
# usage: lane.sh <device> <cpus> <scalar|vector> <mask>...
# Into $ROOT/<scalar|vector>/<mask>.{json,out.pt,run<i>.{json,log}}; $ROOT/lanes.log gets each
# arm's start and end with the device's fuser exit and the CPU set's mpstat idle.
set -u
DEV=$1 CPUS=$2 EXP=$3
shift 3
ROOT=${ROOT:-/mnt/nvme/difflet-exp/sparse/probe}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export PYTHONPATH=$(cd "$HERE/../.." && pwd) NEURON_RT_VISIBLE_CORES=$((4 * DEV)) NEURON_NUMA_PIN=0
FLAG=
[ "$EXP" = vector ] && FLAG=--vector-exp
mkdir -p "$ROOT/$EXP"
state() {
  fuser "/dev/neuron$DEV" > /dev/null 2>&1
  echo "fuser=$? idle%=$(mpstat -P "$CPUS" 1 1 | awk '/^Average/ && $2 ~ /^[0-9]+$/ {s += $NF; n++} END {printf "%.0f", s / n}')"
}
for m in "$@"; do
  name=${m//[:,]/_}
  echo "$(date '+%F %T') start dev=$DEV cpus=$CPUS $EXP $m $(state)" >> "$ROOT/lanes.log"
  taskset -c "$CPUS" "$PY" "$HERE/probe.py" --mask "$m" $FLAG --out "$ROOT/$EXP/$name.json" \
    > "$ROOT/$EXP/$name.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') end   dev=$DEV $EXP $m exit=$rc $(tail -1 "$ROOT/$EXP/$name.log")" >> "$ROOT/lanes.log"
done
