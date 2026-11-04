#!/bin/bash
# Run the TPU profile (layers 2-3) on a v5e-4 node, one job at a time.
#
#   bash benchmark/tpu_profile/run_all.sh [out_dir]
#
# Layer 1 needs no chip: python -m benchmark.tpu_profile.stage_table <run JSONs>.
# Every job gets the chips to itself; a crashed job's leftovers are killed first.
set -u
cd /mnt/models/Difflet
OUT="${1:-/mnt/models/profile/$(date +%Y%m%d-%H%M)}"
mkdir -p "$OUT"
PY=/mnt/models/tt-venv/bin/python
export HF_HOME=/mnt/models/hf

free_chips() {
  c=$(sudo fuser /dev/vfio/* 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | sort -u)
  [ -n "$c" ] && { sudo kill -9 $c; sleep 3; }
  return 0
}
job() {  # $1=label, rest = env assignments then the command
  label="$1"; shift
  free_chips
  echo "== $label ($(date +%H:%M:%S))"
  env "$@" > "$OUT/$label.log" 2>&1
  echo "   exit=$? log=$OUT/$label.log"
}

# The biggest e2e lever per layer 1: the video VAE on a chip. HunyuanVideo's host
# reference takes ~170 s, so it is cached for reruns.
job vae_wan DIFFLET_TPU_COMPILE=0 $PY -m benchmark.tpu_profile.vae_device_probe \
    --model wan_2_1 --dtypes fp32 bf16 --compile \
    --ref-cache "$OUT/ref_wan_2_1.pt" --out "$OUT/vae_wan_2_1.json"
job vae_hv DIFFLET_TPU_COMPILE=0 $PY -m benchmark.tpu_profile.vae_device_probe \
    --model hunyuan_video --dtypes fp32 bf16 \
    --ref-cache "$OUT/ref_hunyuan_video.pt" --out "$OUT/vae_hunyuan_video.json"

# Cheapest first: which tracer works decides how layer 3 is read.
job profiler_probe $PY -m benchmark.tpu_profile.profiler_probe --out-dir "$OUT/trace_probe"

# Layer 2: the block under the configuration that ships (compiled), then eager,
# then compiled with the Pallas flash kernel under self-attention.
job block_compile DIFFLET_TPU_COMPILE=1 $PY -m benchmark.tpu_profile.wan_block_ablation --out "$OUT/block_compile.json" --trace-dir "$OUT/trace_block_compile"
job block_eager   DIFFLET_TPU_COMPILE=0 $PY -m benchmark.tpu_profile.wan_block_ablation --out "$OUT/block_eager.json"
job block_pallas  DIFFLET_TPU_COMPILE=1 DIFFLET_TPU_PALLAS=1 $PY -m benchmark.tpu_profile.wan_block_ablation --out "$OUT/block_pallas.json"

echo DONE > "$OUT/DONE"
