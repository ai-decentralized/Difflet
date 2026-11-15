#!/usr/bin/env bash
# Task 6 phase 2: the timed lanes of one CPU slice, one key after the other (run_arms.sh lane:
# bf16, the quantized arm, linear alone when the plan's rule asks, gen2), each pair on one device
# and this slice from start to finish. Quality is scored elsewhere (no other job of mine on this
# NUMA node while a pair is timed).
# usage: timing_queue.sh <cpus> <key:arm:device>...
set -u
C=$1; shift
WT=/home/ubuntu/Difflet/.claude/worktrees/mxfp8-sage-attention
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
# scratch off the root disk: NKI traces (tempfile) and neuronx-cc work dirs
export TMPDIR=/mnt/nvme/difflet-mxattn/task6/tmp TORCH_NEURONX_DEBUG_DIR=/mnt/nvme/difflet-mxattn/task6/ccwork
# the timeouts HV1.5's cold compile needed (warm_lane3.sh), the same in every arm of every pair
# (controller, 2026-11-15 00:52)
export DIFFLET_NEURON_PG_TIMEOUT_S=10800 TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=10800
export PYTHONPATH=$WT
for e in "$@"; do
  IFS=: read -r K A D <<< "$e"
  echo "$(date '+%F %T') lane $K $A dev $D cpus $C start"
  "$WT/scripts/mx_attention/run_arms.sh" lane "$K" "$A" "$D" "$C"; rc=$?
  echo "$(date '+%F %T') lane $K $A exit $rc"
done
echo TIMING_QUEUE_DONE
