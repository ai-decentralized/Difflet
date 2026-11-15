#!/usr/bin/env bash
# Task 5 device checks (b)/(c), untimed part: one probe process per arm (compile + warm, the
# first output saved, accuracy vs its reference): the fork QK-only with the VectorE exp, flat and
# peaked (rel L2 vs the Task 1 mirror); Difflet's attention() with DIFFLET_ATTN_QUANT=mxfp8
# (--impl routed, flat); shipped attention_cte (--impl cte, flat). From the frozen tree.
# usage: kacc.sh SHAPE DEV CPUS
set -u
S=$1 D=$2 C=$3
B=/mnt/nvme/difflet-mxattn/task5
T=$B/tree
O=$B/kernel
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export PYTHONPATH=$T NEURON_RT_VISIBLE_CORES=$((4 * D)) DIFFLET_NKI_TRACE_CACHE=1
export DIFFLET_COMPILE_CACHE=$B/probe_cache
unset DIFFLET_ATTN_QUANT  # the probe's worker sets it per arm
mkdir -p $O/acc
cd $T
one() {  # NAME FLAGS...
  local n=$1; shift
  {
    echo "$(date '+%F %T') $S $n start dev=$D core=$NEURON_RT_VISIBLE_CORES cpus=$C uptime: $(uptime)"
    fuser /dev/neuron$D; echo "fuser /dev/neuron$D exit $?"
  } >> $O/acc/runs.log 2>&1
  if fuser /dev/neuron$D >/dev/null 2>&1; then echo "$(date '+%F %T') $S BUSY, stop" >> $O/acc/runs.log; exit 3; fi
  timeout 7200 taskset -c "$C" ~/venvs/native-fp8/bin/python scripts/mx_attention/probe.py --shape "$S" "$@" \
    --runs 1 --out $O/acc/$n.$S.json > $O/acc/$n.$S.log 2>&1
  echo "$(date '+%F %T') $S $n end exit=$? uptime: $(uptime)" >> $O/acc/runs.log
}
one fvexp_flat --impl fork --qk-mx --fork-vector-exp
one routed_flat --impl routed
one fvexp_peak --impl fork --qk-mx --fork-vector-exp --peak 2
one cte_flat --impl cte
echo "$(date '+%F %T') $S acc done dev=$D" >> $O/acc/runs.log
