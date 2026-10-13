#!/bin/bash
# Verifier A/B: baseline block NEFFs vs ccop-overlap block NEFFs, g=5, 6 steps, benchmark probe.
set -u
W=/home/ubuntu/Difflet/.claude/worktrees/wan21-bench
V=/home/ubuntu/wan21-bench/verify_meas
PY=/home/ubuntu/workspace/native_venv/bin/python
unset DIFFLET_BACKEND DIFFLET_EXEC_MODE NEURON_RT_NUM_CORES NEURON_LOGICAL_NC_CONFIG TORCH_NEURONX_NEFF_CACHE_DIR TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR NEURON_CC_FLAGS
export PATH=$(dirname $PY):$PATH
export NEURON_RT_VIRTUAL_CORE_SIZE=2 HF_HUB_OFFLINE=1 DIFFLET_NEURON_PG_TIMEOUT_S=3600
export PYTHONPATH=$W/benchmark/wan_step_probe:$W
run() {  # name cache ccop(0/1) extra
  local NAME=$1 CACHE=$2 CCOP=$3 EXTRA=$4
  local D=$V/$NAME; mkdir -p $D/probe; cd $D
  export DIFFLET_COMPILE_CACHE=$CACHE DIFFLET_WAN_PROBE_DIR=$D/probe
  if [ $CCOP = 1 ]; then export NEURON_CC_FLAGS="--tensorizer-options=--enable-ccop-compute-overlap"; else unset NEURON_CC_FLAGS; fi
  echo "START $NAME $(date -u +%s.%N)" >> $V/times.txt
  $PY -m difflet.cli.main generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 --backend neuron --exec-mode compile --cache-dir $CACHE $EXTRA --steps 6 --guidance-scale 5.0 --seed 42 --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4 > $D/log.txt 2>&1
  echo "END $NAME rc=$? $(date -u +%s.%N)" >> $V/times.txt
}
run base1 /home/ubuntu/wan21-bench/neuron/cache 0 ""
run ccop1 /home/ubuntu/wan21-bench/profile/cache_ccop 1 "--host-text-encoder --host-vae"
run base2 /home/ubuntu/wan21-bench/neuron/cache 0 ""
run ccop2 /home/ubuntu/wan21-bench/profile/cache_ccop 1 "--host-text-encoder --host-vae"
echo ALLDONE >> $V/times.txt
