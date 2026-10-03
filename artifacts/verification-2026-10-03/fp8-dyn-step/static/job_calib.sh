#!/usr/bin/env bash
# CPU calibration of static activation scales for Wan 2.1 (real loop, real weights, real prompt).
set -uo pipefail
W=/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
cd "$W"
source .venv/bin/activate
export PYTHONPATH=$PWD DIFFLET_BACKEND=cpu
SNAP=$(ls -d /home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68)
E=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static
mkdir -p "$E"
echo "=== calibrate $(date -u +%FT%TZ) snapshot $SNAP"
python scripts/ptq_calibrate_activations.py --model-dir "$SNAP" --height 480 --width 832 --num-frames 9 --steps 20 \
  --out "$E/act_calibration_wan21.json" 2>&1 | grep -vE 'Warning|warn|import_nki'
echo "CALIB_RC=${PIPESTATUS[0]}"
echo "CALIB_DONE"
