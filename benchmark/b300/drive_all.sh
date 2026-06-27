#!/usr/bin/env bash
# Unified B300 benchmark driver. Runs the full MATRIX serially on the one GPU
# (same protocol as the H100/trn2 reference), with --iters 1 so each model yields
# BOTH e2e cold and one warm e2e iteration (n=1) — matching trn2's warm method.
# Each model's HF cache is pruned after its run to stay within the 422 GB disk.
# HF_TOKEN (for gated flux_1_dev) is read from the environment, never stored here.
set -u
source ~/.venvs/difflet-b300/bin/activate
export HF_HOME=/root/hf
export DIFFLET_BENCH_DEVICE=b300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/Difflet
HUB=/root/hf/hub

# slug | hf-repo (for cache prune) | extra inline env
RUNS=(
  "qwen_image|Qwen/Qwen-Image|"
  "ltx_2|Lightricks/LTX-2|DIFFLET_BENCH_VAE_TILING=0"
  "wan_2_1|Wan-AI/Wan2.1-T2V-14B-Diffusers|"
  "wan_2_2|Wan-AI/Wan2.2-T2V-A14B-Diffusers|"
  "hunyuan_video|hunyuanvideo-community/HunyuanVideo|"
  "hunyuan_video_15|hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v|"
  "flux_1_dev|black-forest-labs/FLUX.1-dev|"
)

prune() { local repo="$1"; local dir="$HUB/models--${repo//\//--}"
  if [ -d "$dir" ]; then echo "[drive] pruning $dir"; rm -rf "$dir"; fi; }

for entry in "${RUNS[@]}"; do
  IFS='|' read -r slug repo extra <<<"$entry"
  log="benchmark/b300/logs/${slug}.log"
  echo "[drive] === $slug (extra='$extra') -> $log ==="; df -h / | tail -1
  if [ -n "$extra" ]; then
    env "$extra" python -m benchmark.bench --backend cuda --model "$slug" --iters 1 >"$log" 2>&1
  else
    python -m benchmark.bench --backend cuda --model "$slug" --iters 1 >"$log" 2>&1
  fi
  echo "[drive] $slug exit=$?"; tail -3 "$log"
  prune "$repo"
done
echo "[drive] ALL_DONE"
