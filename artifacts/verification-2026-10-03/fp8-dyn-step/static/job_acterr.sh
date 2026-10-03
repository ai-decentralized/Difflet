#!/usr/bin/env bash
# CPU per-layer activation-quantization error (dynamic vs static vs weight-only) on the real Wan 2.1 loop.
set -uo pipefail
W=/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
T=/home/ubuntu/.claude/jobs/b5f130d0/tmp
cd "$W"
source .venv/bin/activate
export PYTHONPATH=$PWD DIFFLET_BACKEND=cpu
SNAP=/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68
E=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static
echo "=== act error $(date -u +%FT%TZ)"
python "$T/act_quant_error.py" --model-dir "$SNAP" --calibration "$E/act_calibration_wan21.json" \
  --probe-steps 0,10,19 --threads 8 --out "$E/act_quant_error_cpu.json" 2>&1 | grep -vE 'Warning|warn|import_nki'
echo "ACTERR_RC=${PIPESTATUS[0]}"
echo "ACTERR_DONE"
