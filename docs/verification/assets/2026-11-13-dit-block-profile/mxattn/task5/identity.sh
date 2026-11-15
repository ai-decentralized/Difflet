#!/usr/bin/env bash
# Task 5 device check (a): DIFFLET_ATTN_QUANT unset, one generate with the evidence doc's command
# (docs/verification/2026-11-07-mxfp8-video-models.md section 0 for Qwen-Image --quant mxfp8,
# section 4a / 8h for Wan 2.1 bf16 480x832x9), from the frozen tree. The NEFF/HLO caches are NVMe
# copies of the root caches (a miss writes there, not to the root disk); --cache-dir stays the root
# cache for its quantized directory and weight stores (read only: they hit).
# usage: identity.sh <qwen|wan> <device> <cpus>
set -u
M=$1 D=$2 C=$3
B=/mnt/nvme/difflet-mxattn/task5
T=${TREE:-$B/tree}
O=$B/identity/${TAG:-$M}
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
log "start $M dev=$D cores=$NEURON_RT_VISIBLE_CORES cpus=$C attention.py=$(sha256sum $T/difflet/backends/neuron/ops_impl/attention.py | cut -c1-16) uptime: $(uptime)"
fuser /dev/neuron$D >> $O/runs.log 2>&1; rc=$?
log "fuser /dev/neuron$D exit $rc"
if [ $rc -ne 1 ]; then log "BUSY, not run"; exit 3; fi
df -h / /mnt/nvme >> $O/runs.log
cd $T
taskset -c "$C" ~/venvs/native-fp8/bin/python -m difflet.cli.main generate --backend neuron "${A[@]}" \
  --cache-dir $ROOT --output $O/gen50a.$EXT --work-dir $O/work_a --keep-work-dir > $O/gen50a.log 2>&1
rc=$?
log "exit $rc uptime: $(uptime)"
fuser /dev/neuron$D >> $O/runs.log 2>&1; log "fuser after: exit $?"
md5sum $O/gen50a.$EXT $O/work_a/latents.pt > $O/md5.txt 2>&1
df -h / /mnt/nvme >> $O/runs.log
echo done > $O/done
