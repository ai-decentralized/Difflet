#!/bin/bash
set -u
W=/home/ubuntu/Difflet/.claude/worktrees/wan21-bench
V=/home/ubuntu/wan21-bench/verify_meas
PY=/home/ubuntu/workspace/native_venv/bin/python
CACHE=/home/ubuntu/wan21-bench/neuron/cache
unset DIFFLET_BACKEND DIFFLET_EXEC_MODE NEURON_RT_NUM_CORES NEURON_LOGICAL_NC_CONFIG TORCH_NEURONX_NEFF_CACHE_DIR TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR NEURON_CC_FLAGS OMP_NUM_THREADS
export PATH=$(dirname $PY):$PATH
export NEURON_RT_VIRTUAL_CORE_SIZE=2 HF_HUB_OFFLINE=1 DIFFLET_NEURON_PG_TIMEOUT_S=3600 DIFFLET_COMPILE_CACHE=$CACHE
export PYTHONPATH=/home/ubuntu/wan21-bench/profile/probe2:$W
run() {
  local NAME=$1 OMP=$2
  local D=$V/$NAME; mkdir -p $D/probe2; cd $D
  export DIFFLET_WAN_PROBE2_DIR=$D/probe2
  if [ $OMP != 0 ]; then export OMP_NUM_THREADS=$OMP; else unset OMP_NUM_THREADS; fi
  echo "START $NAME $(date -u +%s.%N)" >> $V/times2.txt
  $PY -m difflet.cli.main generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 --backend neuron --cache-dir $CACHE --exec-mode compile --steps 6 --guidance-scale 5.0 --seed 42 --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4 > $D/log.txt 2>&1
  echo "END $NAME rc=$? $(date -u +%s.%N)" >> $V/times2.txt
}
run omp_default 0
run omp3 3
echo ALLDONE >> $V/times2.txt
