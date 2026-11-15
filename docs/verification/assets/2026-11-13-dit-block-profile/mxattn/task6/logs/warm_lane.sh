#!/usr/bin/env bash
# Task 6 phase 1 (untimed): compile and warm one key's quantized arm, then its bf16 arm, on one
# device; CPU set: the whole NUMA node 1 (no timing runs in this phase).
# usage: warm_lane.sh <key> <quantized arm> <device>
set -u
K=$1 A=$2 D=$3
WT=/home/ubuntu/Difflet/.claude/worktrees/mxfp8-sage-attention
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export PYTHONPATH=$WT
CPUS=48-95,144-191
echo "$(date '+%F %T') warm_lane $K $A dev $D start"
"$WT/scripts/mx_attention/run_arms.sh" warm "$K" "$A" "$D" "$CPUS"; rc=$?
echo "$(date '+%F %T') $K $A warm exit $rc"
"$WT/scripts/mx_attention/run_arms.sh" warm "$K" bf16 "$D" "$CPUS"; rc=$?
echo "$(date '+%F %T') $K bf16 warm exit $rc"
echo WARM_LANE_DONE
