#!/usr/bin/env bash
# Dynamic-activation probe (bf16 + fp8 arms).
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
EVID=$PWD/artifacts/verification-2026-10-01/ptq-wan21/phase0
P=/home/ubuntu/.claude/jobs/b5f130d0/tmp/ptq_probe
echo "=== probe dynamic $(date -u +%FT%TZ)"
python scripts/ptq_fp8_device_probe.py --work-dir "$P" --force-clean 2>&1 | tee "$EVID/probe.log" | grep -vE 'Warning|warnings.warn' | tail -25
rc=${PIPESTATUS[0]}
cp "$P/ptq_probe_report.json" "$EVID/ptq_probe_report.json" 2>/dev/null
echo "PROBE_DYN_RC=$rc"
