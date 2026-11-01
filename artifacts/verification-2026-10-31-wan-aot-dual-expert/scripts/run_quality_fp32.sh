#!/usr/bin/env bash
# quality_fp32.py with the AOT venv on PATH (diffusers' import chain spawns libneuronpjrt-path).
source /home/ubuntu/campaigns/wan-dual-2026-10-31/env.sh
SNAP=~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/$REV
python "$C/quality_fp32.py" "$SNAP" "$C" 2>&1 | grep -v -E "Warning|warn\(" | tail -6
