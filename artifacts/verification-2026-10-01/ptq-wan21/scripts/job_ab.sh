#!/usr/bin/env bash
# Phase 1-3: 14B FP8 vs bf16 A/B (quantize -> compile both -> 2 generates per arm -> compare).
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
OUT=$PWD/artifacts/verification-2026-10-01/ptq-wan21/ab
echo "=== ab start $(date -u +%FT%TZ)"
python scripts/ptq_fp8_ab.py \
  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 --seed 42 \
  --runs 2 --drop-caches --out-dir "$OUT" "$@"
rc=$?
echo "=== ab end $(date -u +%FT%TZ)"
echo "AB_DONE rc=$rc"
