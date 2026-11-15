#!/usr/bin/env bash
# Task 5b device check (a): attention quantization off (no --attn-quant, DIFFLET_ATTN_QUANT unset),
# one generate with the evidence doc's command (Task 5's identity.sh), from the frozen 5b tree.
# NEFF/HLO caches: private NVMe copies of Task 5's (a miss writes there, never to the root disk);
# --cache-dir stays the root cache for its quantized directory and weight stores (read only: hit).
# usage: identity.sh <qwen|wan> <device> <cpus>
set -u
M=$1 D=$2 C=$3
B=/mnt/nvme/difflet-mxattn/task5b
T=$B/tree
O=$B/identity/$M
mkdir -p $O
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export PYTHONPATH=$T HF_HUB_OFFLINE=1
unset DIFFLET_ATTN_QUANT DIFFLET_QUANT_TARGETS DIFFLET_COMPILE_CACHE DIFFLET_NKI_TRACE_CACHE \
  OMP_NUM_THREADS MKL_NUM_THREADS DIFFLET_NEURON_WEIGHT_STORE DIFFLET_SHARED_WEIGHTS_DIR \
  TORCHINDUCTOR_CACHE_DIR
export NEURON_RT_VISIBLE_CORES=$((4 * D))-$((4 * D + 3))
case $M in
  qwen) ROOT=/home/ubuntu/difflet-cache-fp8 N=$B/cache_fp8 EXT=png
        A=(--model-id Qwen/Qwen-Image --tp-degree 4 --height 1024 --width 1024 --steps 50
           --guidance-scale 7.5 --seed 42 --prompt "a cat sitting on a bench" --quant mxfp8) ;;
  wan) ROOT=/home/ubuntu/difflet-cache-video N=$B/cache_video EXT=mp4
       A=(--model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --tp-degree 4 --height 480 --width 832
          --num-frames 9 --steps 50 --seed 42 --prompt "a cat walking through a garden") ;;
  *) echo "unknown model $M" >&2; exit 2 ;;
esac
export TORCH_NEURONX_NEFF_CACHE_DIR=$N/neuron/neff TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=$N/neuron/neff_local \
  TORCH_NEURONX_HLO_CACHE_DIR=$N/neuron/hlo
log() { echo "$(date '+%F %T') $*" >> $O/runs.log; }
log "start $M dev=$D cores=$NEURON_RT_VISIBLE_CORES cpus=$C attention.py=$(sha256sum $T/difflet/backends/neuron/ops_impl/attention.py | cut -c1-16) DIFFLET_ATTN_QUANT=${DIFFLET_ATTN_QUANT:-unset} uptime: $(uptime)"
fuser /dev/neuron$D >> $O/runs.log 2>&1; rc=$?
log "fuser /dev/neuron$D exit $rc"
if [ $rc -ne 1 ]; then log "BUSY, not run"; exit 3; fi
df -h / /mnt/nvme >> $O/runs.log
cd $T
taskset -c "$C" ~/venvs/native-fp8/bin/python -m difflet.cli.main generate --backend neuron "${A[@]}" \
  --cache-dir $ROOT --output $O/gen50.$EXT --work-dir $O/work --keep-work-dir > $O/gen50.log 2>&1
rc=$?
log "exit $rc uptime: $(uptime)"
fuser /dev/neuron$D >> $O/runs.log 2>&1; log "fuser after: exit $?"
md5sum $O/gen50.$EXT $O/work/latents.pt > $O/md5.txt 2>&1
df -h / /mnt/nvme >> $O/runs.log
echo done > $O/done
