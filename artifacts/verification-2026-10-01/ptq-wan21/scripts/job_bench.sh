#!/usr/bin/env bash
# Phase 3b: benchmark harness report files for wan_2_1_fp8 (and the bf16 wan_2_1 cold/warm on this host for like-for-like).
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
EVID=$PWD/artifacts/verification-2026-10-01/ptq-wan21/bench
mkdir -p "$EVID"
echo "=== bench wan_2_1_fp8 $(date -u +%FT%TZ)"
python -m benchmark.bench --model wan_2_1_fp8 --skip-download --skip-compile --iters 1 2>&1 | grep -vE 'Warning|warnings.warn' | tee "$EVID/bench_wan_2_1_fp8.log" | tail -15
echo "BENCH_FP8_RC=${PIPESTATUS[0]}"
echo "=== cold_warm wan_2_1_fp8 $(date -u +%FT%TZ)"
python -m benchmark.cold_warm_e2e --model wan_2_1_fp8 2>&1 | grep -vE 'Warning|warnings.warn' | tee "$EVID/cold_warm_wan_2_1_fp8.log" | tail -8
echo "COLDWARM_FP8_RC=${PIPESTATUS[0]}"
echo "=== bench wan_2_1_fp8_wo $(date -u +%FT%TZ)"
python -m benchmark.bench --model wan_2_1_fp8_wo --skip-download --skip-compile --iters 1 2>&1 | grep -vE 'Warning|warnings.warn' | tee "$EVID/bench_wan_2_1_fp8_wo.log" | tail -15
echo "BENCH_FP8WO_RC=${PIPESTATUS[0]}"
echo "=== cold_warm wan_2_1_fp8_wo $(date -u +%FT%TZ)"
python -m benchmark.cold_warm_e2e --model wan_2_1_fp8_wo 2>&1 | grep -vE 'Warning|warnings.warn' | tee "$EVID/cold_warm_wan_2_1_fp8_wo.log" | tail -8
echo "COLDWARM_FP8WO_RC=${PIPESTATUS[0]}"
echo "=== cold_warm wan_2_1 (bf16, same host) $(date -u +%FT%TZ)"
python -m benchmark.cold_warm_e2e --model wan_2_1 2>&1 | grep -vE 'Warning|warnings.warn' | tee "$EVID/cold_warm_wan_2_1.log" | tail -8
echo "COLDWARM_BF16_RC=${PIPESTATUS[0]}"
echo "BENCH_DONE"
