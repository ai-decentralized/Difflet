#!/usr/bin/env bash
# Task 6 fix round 1, ruling BR37: the fork with allow_vector_exp=True and no MX flags
# (probe --impl fork --fork-vector-exp) against shipped attention_cte (probe --impl cte, the VectorE
# exp), one probe process each on one logical core, first output saved; bit identity: cmp.py.
# usage: ident.sh SHAPE ARM DEV CPUS   (ARM: cte | fvexp)
set -u
S=$1 A=$2 D=$3 C=$4
WT=/home/ubuntu/Difflet/.claude/worktrees/mxfp8-sage-attention
O=/mnt/nvme/difflet-mxattn/task6/ident
case $A in
  cte) F=(--impl cte) ;;
  fvexp) F=(--impl fork --fork-vector-exp) ;;
  *) echo "unknown arm $A" >&2; exit 2 ;;
esac
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export PYTHONPATH=$WT NEURON_RT_VISIBLE_CORES=$((4 * D)) DIFFLET_NKI_TRACE_CACHE=1
export DIFFLET_COMPILE_CACHE=$O/cache_d$D  # one per lane: no concurrent writers
export TMPDIR=/mnt/nvme/difflet-mxattn/task6/tmp TORCH_NEURONX_DEBUG_DIR=/mnt/nvme/difflet-mxattn/task6/ccwork
unset DIFFLET_ATTN_QUANT
cd "$WT" || exit 1
{
  echo "$(date '+%F %T') $S $A start dev=$D core=$NEURON_RT_VISIBLE_CORES cpus=$C uptime: $(uptime)"
  fuser /dev/neuron$D; echo "fuser /dev/neuron$D exit $?"
} >> $O/runs.log 2>&1
if fuser /dev/neuron$D >/dev/null 2>&1; then echo "$(date '+%F %T') $S $A BUSY, stop" >> $O/runs.log; exit 3; fi
timeout 7200 taskset -c "$C" ~/venvs/native-fp8/bin/python scripts/mx_attention/probe.py --shape "$S" \
  "${F[@]}" --runs 1 --out $O/$A.$S.json > $O/$A.$S.log 2>&1
echo "$(date '+%F %T') $S $A end exit=$? uptime: $(uptime)" >> $O/runs.log
