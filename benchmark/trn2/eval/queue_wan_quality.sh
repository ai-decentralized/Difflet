#!/usr/bin/env bash
# Wan 2.1 step-caching quality on the evaluated program (LNC2 attention, 81 frames, 50
# steps, CFG 5.0, chunked FP32 VAE on the device), filling the paper's missing Wan row:
# 16 VBench prompts (scripts/data/vbench_overall_consistency_first16.txt), seed 42,
# caching off (reference), cadence 2, fewer steps (30 of 50), online-delta (alpha 0.6).
# Calibrated adaptive is left out: its probe program is not compiled for this program.
# Prompt-major and resumable: rerun the same command to continue; every finished prompt
# is committed and pushed (manifest.jsonl, progress.json).
#
# Results: benchmark/trn2-eval-vc2/quality/wan_2_1/
#
#   setsid nohup benchmark/trn2/eval/queue_wan_quality.sh > LOG 2>&1 &   (venv active)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE=trn2-eval-vc2
PY="${PYTHON_BIN:-python}"
echo "start $(date -u +%FT%TZ) $(git rev-parse --short HEAD)"
"$PY" -m benchmark.caching_quality run --model wan_2_1 \
    --configs tp4 tp4tc2 tp4s30 tp4tcod \
    --prompts scripts/data/vbench_overall_consistency_first16.txt --git-sync
echo "end $(date -u +%FT%TZ) exit=$?"
