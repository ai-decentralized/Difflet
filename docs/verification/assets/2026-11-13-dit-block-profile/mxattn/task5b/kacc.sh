#!/usr/bin/env bash
# Task 5b device check (b): per variant, Difflet's attention() under DIFFLET_ATTN_QUANT=<variant>
# (--impl routed with the variant's flags) and the fork with the same flags and the VectorE exp
# (--impl fork --fork-vector-exp), one probe process each (compile + warm, the first output saved,
# accuracy vs that variant's Task 1 mirror). From the frozen 5b tree. Bit identity: cmp.py.
# usage: kacc.sh SHAPE VARIANT DEV CPUS
set -u
S=$1 V=$2 D=$3 C=$4
B=/mnt/nvme/difflet-mxattn/task5b
T=$B/tree
O=$B/kernel
case $V in
  qk) F=(--qk-mx) ;;
  pv) F=(--pv-mx) ;;
  qk+pv) F=(--qk-mx --pv-mx) ;;
  *) echo "unknown variant $V" >&2; exit 2 ;;
esac
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export PYTHONPATH=$T NEURON_RT_VISIBLE_CORES=$((4 * D)) DIFFLET_NKI_TRACE_CACHE=1
export DIFFLET_COMPILE_CACHE=$B/probe_cache_d$D  # one per lane: no concurrent writers
unset DIFFLET_ATTN_QUANT  # the probe's worker sets it per arm
mkdir -p $O
cd $T
one() {  # NAME FLAGS...
  local n=$1; shift
  {
    echo "$(date '+%F %T') $S $V $n start dev=$D core=$NEURON_RT_VISIBLE_CORES cpus=$C uptime: $(uptime)"
    fuser /dev/neuron$D; echo "fuser /dev/neuron$D exit $?"
  } >> $O/runs.log 2>&1
  if fuser /dev/neuron$D >/dev/null 2>&1; then echo "$(date '+%F %T') $S $V BUSY, stop" >> $O/runs.log; exit 3; fi
  timeout 7200 taskset -c "$C" ~/venvs/native-fp8/bin/python scripts/mx_attention/probe.py --shape "$S" "$@" \
    --runs 1 --out $O/$n.$V.$S.json > $O/$n.$V.$S.log 2>&1
  echo "$(date '+%F %T') $S $V $n end exit=$? uptime: $(uptime)" >> $O/runs.log
}
one routed --impl routed "${F[@]}"
one fvexp --impl fork "${F[@]}" --fork-vector-exp
echo "$(date '+%F %T') $S $V done dev=$D" >> $O/runs.log
