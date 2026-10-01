#!/usr/bin/env bash
# Wait for the main sweep (t=900, t=100 at the 240 range) to finish, then rerun t=500 at the 240 range.
set -uo pipefail
until [ -f /home/ubuntu/.claude/jobs/b5f130d0/tmp/logs/sweep.done ]; do sleep 30; done
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
SNAP=/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68
OUT=$PWD/artifacts/verification-2026-10-01/ptq-wan21/linear_sweep
mv "$OUT/linear_error_t500.json" "$OUT/linear_error_t500_fp8max448_superseded.json"
mv "$OUT/sweep_t500.log" "$OUT/sweep_t500_fp8max448_superseded.log"
echo "=== sweep t=500 rerun (FP8_MAX=240) $(date -u +%FT%TZ)"
python scripts/ptq_linear_error_sweep.py --model-dir "$SNAP" --height 480 --width 832 --num-frames 9 \
  --timestep 500 --m-slice 1024 --out "$OUT/linear_error_t500.json" 2>&1 | tee "$OUT/sweep_t500.log" | tail -12
echo "SWEEP_T500_RERUN_RC=${PIPESTATUS[0]}"
