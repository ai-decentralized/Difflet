#!/usr/bin/env bash
# Task 5b device checks (d) and (c, without --quant): Qwen-Image 1024x1024 tp4 bf16 linears (no
# --quant). `difflet compile --attn-quant pv`, then `difflet compile --attn-quant qk`, into ONE
# cache root R (--cache-dir R, DIFFLET_COMPILE_CACHE unset: NEFF/HLO caches under R/neuron, stage
# manifests under R/qwen_image_dit/<key>), then a short `generate --attn-quant pv` (STEPS steps)
# from R. Between steps: the stage dirs and the NEFF cache entries (no reuse = the second variant
# compiles new NEFFs into a second stage dir). From the frozen 5b tree.
# usage: droot.sh <device> <cpus> <steps>
set -u
D=$1 C=$2 STEPS=$3
B=/mnt/nvme/difflet-mxattn/task5b
T=$B/tree
O=$B/droot
R=$B/droot/root
mkdir -p $O $R
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export PYTHONPATH=$T HF_HUB_OFFLINE=1
unset DIFFLET_ATTN_QUANT DIFFLET_QUANT_TARGETS DIFFLET_COMPILE_CACHE DIFFLET_NKI_TRACE_CACHE \
  OMP_NUM_THREADS MKL_NUM_THREADS DIFFLET_NEURON_WEIGHT_STORE DIFFLET_SHARED_WEIGHTS_DIR \
  TORCHINDUCTOR_CACHE_DIR TORCH_NEURONX_NEFF_CACHE_DIR TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR \
  TORCH_NEURONX_HLO_CACHE_DIR
export NEURON_RT_VISIBLE_CORES=$((4 * D))-$((4 * D + 3))
A=(--backend neuron --model-id Qwen/Qwen-Image --tp-degree 4 --height 1024 --width 1024
   --cache-dir $R)
log() { echo "$(date '+%F %T') $*" >> $O/runs.log; }
snap() {  # what the cache root holds: stage dirs (with their attn_quant) and NEFF entries
  log "snapshot $1: stage dirs: $(cd $R && ls -d qwen_image_dit/* 2>/dev/null | tr '\n' ' ')"
  for m in $R/qwen_image_dit/*/manifest.json; do
    [ -f "$m" ] && log "  $m attn_quant=$(~/venvs/native-fp8/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['cache_inputs'].get('attn_quant'))" $m)"
  done
  (cd $R/neuron 2>/dev/null && find . -name '*.neff' | sort) > $O/neffs.$1.txt
  log "  neff files: $(wc -l < $O/neffs.$1.txt)"
}
step() {  # NAME CMD...
  local n=$1; shift
  log "start $n dev=$D cores=$NEURON_RT_VISIBLE_CORES cpus=$C shell DIFFLET_ATTN_QUANT=${DIFFLET_ATTN_QUANT:-unset} uptime: $(uptime)"
  fuser /dev/neuron$D >> $O/runs.log 2>&1; rc=$?
  log "fuser /dev/neuron$D exit $rc"
  if [ $rc -ne 1 ]; then log "BUSY, stop"; exit 3; fi
  df -h / /mnt/nvme >> $O/runs.log
  (cd $T && taskset -c "$C" ~/venvs/native-fp8/bin/python -m difflet.cli.main "$@" > $O/$n.log 2>&1)
  rc=$?
  log "end $n exit $rc uptime: $(uptime)"
  fuser /dev/neuron$D >> $O/runs.log 2>&1; log "fuser after: exit $?"
  [ $rc -eq 0 ] || exit 4
}
log "attention.py=$(sha256sum $T/difflet/backends/neuron/ops_impl/attention.py | cut -c1-16)"
snap 0_empty
step compile_pv compile "${A[@]}" --attn-quant pv
snap 1_after_pv
step compile_qk compile "${A[@]}" --attn-quant qk
snap 2_after_qk
step generate_pv generate "${A[@]}" --attn-quant pv --steps "$STEPS" --guidance-scale 7.5 \
  --seed 42 --prompt "a cat sitting on a bench" --output $O/gen_pv.png --work-dir $O/work_pv \
  --keep-work-dir
snap 3_after_generate_pv
md5sum $O/gen_pv.png $O/work_pv/latents.pt > $O/md5.txt 2>&1
df -h / /mnt/nvme >> $O/runs.log
echo done > $O/done
