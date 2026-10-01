#!/usr/bin/env bash
# CPU full-forward numerics (fp32 / bf16 / fp8-wo / fp8-dyn) on the real 14B weights, after the device work.
set -uo pipefail
until [ -f /home/ubuntu/.claude/jobs/b5f130d0/tmp/logs/probe_fixed.done ]; do sleep 30; done
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD DIFFLET_BACKEND=cpu
SNAP=/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68
OUT=$PWD/artifacts/verification-2026-10-01/ptq-wan21/full_forward
mkdir -p "$OUT"
echo "=== full forward t=500 $(date -u +%FT%TZ)"
python scripts/ptq_full_forward_error.py --model-dir "$SNAP" --height 480 --width 832 --num-frames 9 --timestep 500 \
  --out "$OUT/full_forward_error_t500.json" 2>&1 | grep -vE 'Warning|warnings.warn' | tee "$OUT/full_forward_t500.log" | tail -12
echo "FULLFWD_RC=${PIPESTATUS[0]}"
echo "FULLFWD_DONE"
