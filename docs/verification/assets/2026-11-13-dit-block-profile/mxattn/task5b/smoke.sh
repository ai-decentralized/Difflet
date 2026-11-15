#!/usr/bin/env bash
# Task 5b device check (c): Qwen-Image 1024x1024 tp4 --quant mxfp8, a short generate (STEPS steps,
# seed 42, the section-0 prompt and guidance) through the CLI flag --attn-quant VARIANT, with
# DIFFLET_ATTN_QUANT unset in the shell (the flag alone sets it). From the frozen 5b tree; NEFF/HLO
# cache: a private NVMe copy of Task 5's smoke cache per variant; --cache-dir the root cache
# (quantized dir, read only; the neuron generate stage needs no stage manifest).
# usage: smoke.sh <qk|pv|qk+pv> <device> <cpus> <steps>
set -u
V=$1 D=$2 C=$3 STEPS=$4
B=/mnt/nvme/difflet-mxattn/task5b
T=$B/tree
O=$B/smoke/quant_$V
N=$B/cache_smoke_$V
mkdir -p $O
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export PYTHONPATH=$T HF_HUB_OFFLINE=1
unset DIFFLET_ATTN_QUANT DIFFLET_QUANT_TARGETS DIFFLET_COMPILE_CACHE DIFFLET_NKI_TRACE_CACHE \
  OMP_NUM_THREADS MKL_NUM_THREADS DIFFLET_NEURON_WEIGHT_STORE DIFFLET_SHARED_WEIGHTS_DIR \
  TORCHINDUCTOR_CACHE_DIR
export NEURON_RT_VISIBLE_CORES=$((4 * D))-$((4 * D + 3))
export TORCH_NEURONX_NEFF_CACHE_DIR=$N/neuron/neff TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=$N/neuron/neff_local \
  TORCH_NEURONX_HLO_CACHE_DIR=$N/neuron/hlo
log() { echo "$(date '+%F %T') $*" >> $O/runs.log; }
log "start --attn-quant $V --quant mxfp8 steps=$STEPS dev=$D cores=$NEURON_RT_VISIBLE_CORES cpus=$C shell DIFFLET_ATTN_QUANT=${DIFFLET_ATTN_QUANT:-unset} attention.py=$(sha256sum $T/difflet/backends/neuron/ops_impl/attention.py | cut -c1-16) uptime: $(uptime)"
fuser /dev/neuron$D >> $O/runs.log 2>&1; rc=$?
log "fuser /dev/neuron$D exit $rc"
if [ $rc -ne 1 ]; then log "BUSY, not run"; exit 3; fi
df -h / /mnt/nvme >> $O/runs.log
cd $T
taskset -c "$C" ~/venvs/native-fp8/bin/python -m difflet.cli.main generate --backend neuron \
  --model-id Qwen/Qwen-Image --tp-degree 4 --height 1024 --width 1024 --steps "$STEPS" \
  --guidance-scale 7.5 --seed 42 --prompt "a cat sitting on a bench" --quant mxfp8 \
  --attn-quant "$V" --cache-dir /home/ubuntu/difflet-cache-fp8 --output $O/gen.png \
  --work-dir $O/work --keep-work-dir > $O/gen.log 2>&1
rc=$?
log "exit $rc uptime: $(uptime)"
fuser /dev/neuron$D >> $O/runs.log 2>&1; log "fuser after: exit $?"
md5sum $O/gen.png $O/work/latents.pt > $O/md5.txt 2>&1
df -h / /mnt/nvme >> $O/runs.log
echo done > $O/done
