#!/usr/bin/env bash
# Task 6 phase 1, third form (HV1.5, again): warm_lane.sh with the scratch on NVMe and the
# NEFF-cache lock timeout raised to 3 h, so ranks waiting on another rank's cold compile keep
# waiting instead of breaking the lock after 30 min, compiling the same NEFF again and leaving the
# first rank alone in a collective until the 1800 s watchdog (the first HV1.5 attempt, 22:51-23:04).
# usage: warm_lane2.sh <key> <quantized arm> <device>
set -u
K=$1 A=$2 D=$3
WT=/home/ubuntu/Difflet/.claude/worktrees/mxfp8-sage-attention
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export PYTHONPATH=$WT
export TMPDIR=/mnt/nvme/difflet-mxattn/task6/tmp TORCH_NEURONX_DEBUG_DIR=/mnt/nvme/difflet-mxattn/task6/ccwork
export TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=10800
# and the process group timeout: a block compiled cold inside the first collective took over
# the 1800 s default (HV1.5 attn, 00:26)
export DIFFLET_NEURON_PG_TIMEOUT_S=10800
CPUS=48-95,144-191
echo "$(date '+%F %T') warm_lane3 $K $A dev $D start"
"$WT/scripts/mx_attention/run_arms.sh" warm "$K" "$A" "$D" "$CPUS"; rc=$?
echo "$(date '+%F %T') $K $A warm exit $rc"
"$WT/scripts/mx_attention/run_arms.sh" warm "$K" bf16 "$D" "$CPUS"; rc=$?
echo "$(date '+%F %T') $K bf16 warm exit $rc"
echo WARM_LANE_DONE
