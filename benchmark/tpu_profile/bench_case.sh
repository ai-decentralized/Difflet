#!/bin/bash
# One case of main's bench on a TPU VM, as the A/B tables in
# docs/plans/2026-11-10-tpu-collective-matmul-pallas.md were measured.
#
#   benchmark/tpu_profile/bench_case.sh <tag> <model> [ENV=VALUE ...]
#
# e.g. (Wan, padded SP vs not, back to back on one node):
#   W="DIFFLET_TPU_COMPILE=1 DIFFLET_TPU_DEVICE_VAE=1 DIFFLET_TPU_SP=1"
#   benchmark/tpu_profile/bench_case.sh wp-pad0 wan_2_1 $W DIFFLET_TPU_WAN_SP_PAD=0
#   benchmark/tpu_profile/bench_case.sh wp-pad1 wan_2_1 $W DIFFLET_TPU_WAN_SP_PAD=1
#   python -m benchmark.tpu_profile.image_psnr $OUT/wp-pad0/<frames>.png $OUT/wp-pad1/<frames>.png
#
# Add DIFFLET_BENCH_TRACE_DIR=<dir> for a jax.profiler trace (then
# benchmark.tpu_profile.trace_diff / exposed_comm). Frees the chips first (a
# previous case may still hold /dev/vfio), runs `benchmark.bench` once
# (--iters 1 --natural-iters 1: the step time is the mean over the request's
# denoising steps), saves the outputs under $OUT/<tag> and the log as
# $OUT/<tag>.log, and prints one summary line.
#
# Paths default to the layout of the data disk used on 2026-11-11 (torch 2.14.1+cpu,
# torch_tpu 0.2.0, libtpu 0.0.51, jax 0.12.0.1 in $VENV); override as needed.
set -u
tag=$1; model=$2; shift 2
ROOT=${DIFFLET_ROOT:-/mnt/models/Difflet-main}
VENV=${DIFFLET_TPU_VENV:-/mnt/models/tt-venv}
OUT=${DIFFLET_BENCH_OUT:-/mnt/models/profile}
cd "$ROOT"
holders=$(sudo fuser /dev/vfio/* 2>/dev/null | tr -s " " "\n" | grep -E "^[0-9]+$" | sort -u)
[ -n "$holders" ] && { sudo kill -9 $holders; sleep 3; }
export PYTHONPATH=$ROOT HF_HOME=${HF_HOME:-/mnt/models/hf} DIFFLET_BACKEND=tpu
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/mnt/models/uvpy/cpython-3.12.14-linux-x86_64-gnu/lib}
export DIFFLET_BENCH_DEVICE=v5e-$tag DIFFLET_BENCH_SAVE_DIR=$OUT/$tag
for kv in "$@"; do export "$kv"; done
echo "tree md5 $(find difflet -name '*.py' | sort | xargs md5sum | md5sum | cut -c1-12)" > "$OUT/$tag.log"
timeout 2400 "$VENV/bin/python" -m benchmark.bench --model "$model" --backend tpu \
    --iters 1 --natural-iters 1 >> "$OUT/$tag.log" 2>&1
status=$?
step=$("$VENV/bin/python" -c "import json; d=json.load(open('benchmark/v5e-$tag/$model.json')); print(round(d['step_latency']['mean'], 4))" 2>/dev/null)
echo "== $tag exit=$status $(grep -oE 'status=[a-z_]+' "$OUT/$tag.log" | tail -1) step=$step" \
     "files=$(ls "$OUT/$tag" 2>/dev/null | tr '\n' ' ')"
