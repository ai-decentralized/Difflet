#!/usr/bin/env bash
# Phase 2a: CPU per-linear fp8 matmul error sweep on real Wan 2.1 14B weights, three timesteps.
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
SNAP=/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68
OUT=$PWD/artifacts/verification-2026-10-01/ptq-wan21/linear_sweep
mkdir -p "$OUT"
for t in 500 900 100; do
  echo "=== sweep t=$t $(date -u +%FT%TZ)"
  python scripts/ptq_linear_error_sweep.py --model-dir "$SNAP" --height 480 --width 832 --num-frames 9 \
    --timestep $t --m-slice 1024 --out "$OUT/linear_error_t$t.json" 2>&1 | tee "$OUT/sweep_t$t.log" | tail -25
  echo "SWEEP_T${t}_RC=${PIPESTATUS[0]}"
done
echo "SWEEP_DONE"
