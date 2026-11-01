#!/usr/bin/env bash
source /home/ubuntu/campaigns/wan-dual-2026-10-31/env.sh
python "$C/fp32_sheet.py" "$C/out/fp32_bf16_aot_dual.png" 2>&1 | grep -v -E "Warning|warn\(" | tail -2
