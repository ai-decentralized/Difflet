#!/bin/bash
# usage: run.sh <neuron|trainium> <name> <steps> <guidance>   (extra env via environment, e.g. PROBE2_SEGMENTS=1, INSPECT=1)
set -u
B=$1; NAME=$2; STEPS=$3; G=$4
W=/home/ubuntu/Difflet/.claude/worktrees/wan21-bench
P=/home/ubuntu/wan21-bench/profile
D=$P/runs/$B-$NAME
mkdir -p $D/probe2
cd $D
unset DIFFLET_BACKEND DIFFLET_EXEC_MODE NEURON_RT_NUM_CORES NEURON_LOGICAL_NC_CONFIG TORCH_NEURONX_NEFF_CACHE_DIR TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR
if [ $B = neuron ]; then PY=/home/ubuntu/workspace/native_venv/bin/python; EXTRA="--exec-mode compile"; else PY=/home/ubuntu/Difflet/.venv/bin/python; EXTRA=""; fi
CACHE=${CACHE_OVERRIDE:-/home/ubuntu/wan21-bench/$B/cache}
export PATH=$(dirname $PY):$PATH
export NEURON_RT_VIRTUAL_CORE_SIZE=2 HF_HUB_OFFLINE=1 DIFFLET_NEURON_PG_TIMEOUT_S=3600 DIFFLET_COMPILE_CACHE=$CACHE
export PYTHONPATH=$P/probe2:$W
export DIFFLET_WAN_PROBE2_DIR=$D/probe2
[ -n "${EXTRA_ARGS:-}" ] && EXTRA="$EXTRA $EXTRA_ARGS"
[ "${PROBE2_SEGMENTS:-0}" = 1 ] && export DIFFLET_WAN_PROBE2_SEGMENTS=1
if [ "${INSPECT:-0}" = 1 ]; then
  export NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_OUTPUT_DIR=$D/inspect NEURON_RT_INSPECT_SYSTEM_PROFILE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=${INSPECT_DEVICE:-0}
  mkdir -p $D/inspect
fi
grep -E '^(Cached|MemFree):' /proc/meminfo > $D/meminfo_start.txt
echo "START $(date -u +%s.%N)" > $D/times.txt
/usr/bin/time -v $PY -m difflet.cli.main generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 --backend $B --cache-dir $CACHE $EXTRA --steps $STEPS --guidance-scale $G --seed 42 --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4 > $D/log.txt 2>&1
echo "RC $?" >> $D/times.txt
echo "END $(date -u +%s.%N)" >> $D/times.txt
