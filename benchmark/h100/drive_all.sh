#!/usr/bin/env bash
# Unified H100 benchmark driver. Runs the full MATRIX serially on the one GPU
# (same protocol + toolchain as the B300/trn2 reference). Each model is measured in
# TWO processes so the 80 GB H100 doesn't OOM:
#   1. `bench --iters 0`             -> cold e2e + per-step + peak mem + output (proc A)
#   2. `warm_e2e --backend cuda`     -> one WARM-cache generate, n=1 (proc B), patched
#      into the same json. (B300 did both in one process via `--iters 1`; its 275 GB
#      had headroom. On H100 the warm iter must be its own process — the cold run's
#      allocator memory isn't released in time to fit a second in-process generate.
#      The number is the equivalent: a second from_pretrained+generate, disk cache warm.)
# Weights cache on the 700 GB /ephemeral scratch (no prune — the whole matrix fits).
# HF_TOKEN (for gated flux_1_dev) is read from the environment, never stored here.
set -u
source ~/.venvs/difflet-h100/bin/activate
export HF_HOME=/ephemeral/hf
export DIFFLET_BENCH_DEVICE=h100
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/Difflet

# slug | extra inline env
RUNS=(
  "qwen_image|"
  "ltx_2|DIFFLET_BENCH_VAE_TILING=0"
  "wan_2_1|"
  "wan_2_2|"
  "hunyuan_video|"
  "hunyuan_video_15|"
  "flux_1_dev|"
)

for entry in "${RUNS[@]}"; do
  IFS='|' read -r slug extra <<<"$entry"
  log="benchmark/h100/logs/${slug}.log"
  echo "[drive] === $slug (extra='$extra') -> $log ==="; df -h /ephemeral | tail -1
  : >"$log"
  # 1) cold + per-step + mem
  env ${extra:+$extra} python -m benchmark.bench --backend cuda --model "$slug" --iters 0 >>"$log" 2>&1
  cold_rc=$?
  echo "[drive] $slug cold exit=$cold_rc"
  # 2) warm (separate process) — only if cold succeeded
  if [ $cold_rc -eq 0 ] && grep -q '"status": "ok"' "benchmark/h100/${slug}.json" 2>/dev/null; then
    env ${extra:+$extra} python -m benchmark.warm_e2e --backend cuda --model "$slug" --iters 1 >>"$log" 2>&1
    echo "[drive] $slug warm exit=$?"
  else
    echo "[drive] $slug: cold failed/non-ok -> skipping warm"
  fi
  tail -3 "$log"
done
echo "[drive] ALL_DONE"
