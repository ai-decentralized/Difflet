#!/usr/bin/env bash
# Task 5 device check (c), timed (ruling BR32): one shape, three arms on one device and CPU set in
# one window, three processes per arm, the order rotated per round (4b's trio.sh method):
# (cte, fvexp, routed), (fvexp, routed, cte), (routed, cte, fvexp). cte = shipped attention_cte
# (--impl cte, the VectorE exp the bf16 arm runs), fvexp = the fork QK-only with the VectorE exp
# (--impl fork --qk-mx --fork-vector-exp), routed = Difflet's attention() under
# DIFFLET_ATTN_QUANT=mxfp8 (--impl routed). Each process is probe.py's worker (fresh runtime, cache
# hit, one warm call, CALLS timed calls, one sync); run 0 of each arm saves its output. fuser +
# uptime before and after every process; stops if the device is busy.
# usage: ktrio.sh SHAPE DEV CPUS CALLS
set -u
S=$1 D=$2 C=$3 N=$4
B=/mnt/nvme/difflet-mxattn/task5
T=$B/tree
O=$B/kernel/trio
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export PYTHONPATH=$T NEURON_RT_VISIBLE_CORES=$((4 * D)) DIFFLET_NKI_TRACE_CACHE=1
export DIFFLET_COMPILE_CACHE=$B/probe_cache
unset DIFFLET_ATTN_QUANT
mkdir -p $O/cte $O/fvexp $O/routed
cd $T
arm() {  # ARM RUN
  local a=$1 i=$2 flags save=""
  case $a in
    cte) flags="--impl cte" ;;
    fvexp) flags="--impl fork --qk-mx --fork-vector-exp" ;;
    routed) flags="--impl routed" ;;
  esac
  [ "$i" = 0 ] && save=--save-output
  {
    echo "$(date '+%F %T') $S $a run$i start dev=$D core=$NEURON_RT_VISIBLE_CORES cpus=$C calls=$N attention.py=$(sha256sum $T/difflet/backends/neuron/ops_impl/attention.py | cut -c1-16) kernel=$(sha256sum $T/difflet/backends/neuron/ops_impl/mx_attention_kernel.py | cut -c1-16)"
    echo "uptime: $(uptime)"
    fuser /dev/neuron$D; echo "fuser /dev/neuron$D exit $?"
  } >> $O/runs.log 2>&1
  if fuser /dev/neuron$D >/dev/null 2>&1; then echo "$(date '+%F %T') $S BUSY, stop" >> $O/runs.log; exit 3; fi
  timeout 3600 taskset -c "$C" ~/venvs/native-fp8/bin/python scripts/mx_attention/probe.py --shape "$S" $flags \
    --out $O/$a/$S.json --calls "$N" --worker $O/$a/$S.run$i.json $save > $O/$a/$S.run$i.log 2>&1
  local rc=$?
  { echo "$(date '+%F %T') $S $a run$i end exit=$rc uptime: $(uptime)"; cat $O/$a/$S.run$i.json 2>/dev/null; echo; } >> $O/runs.log 2>&1
  [ $rc -eq 0 ] || exit $rc
}
arm cte 0; arm fvexp 0; arm routed 0
arm fvexp 1; arm routed 1; arm cte 1
arm routed 2; arm cte 2; arm fvexp 2
echo "$(date '+%F %T') $S trio done dev=$D" >> $O/runs.log
