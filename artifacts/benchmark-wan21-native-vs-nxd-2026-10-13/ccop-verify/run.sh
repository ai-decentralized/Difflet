#!/bin/bash
# On-device verification of the ccop-overlap fix on the neuron backend (worktree HEAD + fix):
# a fresh cache (the flag is part of every NEFF's cache key), then the benchmark's g=5 / g=1
# protocol (first, warm x3 g=5, warm x2 g=1) and a 2-step g=5 run for the fp32-oracle parity.
set -u
W=/home/ubuntu/Difflet/.claude/worktrees/wan21-bench
B=/home/ubuntu/wan21-bench/ccop-verify
CACHE=$B/cache
PY=/home/ubuntu/workspace/native_venv/bin/python
unset NEURON_CC_FLAGS DIFFLET_NEURON_CCOP_OVERLAP NEURON_RT_NUM_CORES NEURON_LOGICAL_NC_CONFIG DIFFLET_BACKEND
export PATH=/home/ubuntu/workspace/native_venv/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_VIRTUAL_CORE_SIZE=2 HF_HUB_OFFLINE=1 \
  DIFFLET_NEURON_PG_TIMEOUT_S=3600 DIFFLET_COMPILE_CACHE=$CACHE PYTHONPATH=$W/benchmark/wan_step_probe:$W
run() {  # name steps guidance
  local d=$B/$1; mkdir -p $d; cd $d
  local t0=$(date +%s.%N)
  DIFFLET_WAN_PROBE_DIR=$d/probe $PY -m difflet.cli.main generate \
    --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 --backend neuron \
    --cache-dir $CACHE --exec-mode compile --steps $2 --guidance-scale $3 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4 > log.txt 2>&1
  local rc=$?
  echo "$1 rc=$rc wall=$(echo "$(date +%s.%N) - $t0" | bc)" | tee -a $B/walls.txt
}
run first-g5 50 5.0
run warm1-g5 50 5.0
run warm2-g5 50 5.0
run warm3-g5 50 5.0
run warm1-g1 50 1.0
run warm2-g1 50 1.0
run twostep-g5 2 5.0
echo done > $B/DONE
