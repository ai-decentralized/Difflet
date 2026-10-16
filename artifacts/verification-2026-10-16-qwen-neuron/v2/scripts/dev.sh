#!/bin/bash
# usage: dev.sh <cache_root> <preflight args...>   (4 ranks, one device job at a time)
WT=/home/ubuntu/Difflet/.claude/worktrees/qwen-image-neuron
VENV=/home/ubuntu/workspace/native_venv
SNAP=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6
cache="$1"; shift
echo "launch $(date -u +%FT%TZ) head=$(git -C $WT rev-parse --short HEAD) cache=$cache args: $*"
start=$(date +%s.%N)
# The venv's bin/ on PATH (neuronx-cc is spawned from there), the cache layout of
# difflet.backends.neuron.compile.neuron_cache_env exported before torch-neuronx is imported.
env PATH=$VENV/bin:$PATH HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3 DIFFLET_NO_JEMALLOC=1 \
  DIFFLET_DISABLE_PREWARM=1 DIFFLET_NEURON_PG_TIMEOUT_S=7200 DIFFLET_COMPILE_CACHE="$cache" \
  TORCH_NEURONX_NEFF_CACHE_DIR="$cache/neuron/neff" \
  TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR="$cache/neuron/neff_local" \
  TORCH_NEURONX_HLO_CACHE_DIR="$cache/neuron/hlo" PYTHONPATH=$WT \
  $VENV/bin/torchrun --standalone --nproc-per-node 4 \
  $WT/tests/manual/check_neuron_qwen_preflight.py --model-dir $SNAP "$@"
rc=$?
end=$(date +%s.%N)
echo "EXIT rc=$rc launch_wall_s=$(echo "$end - $start" | bc) $(date -u +%FT%TZ)"
