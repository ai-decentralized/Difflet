#!/usr/bin/env bash
# Phase 0b attempt 3: dynamic-activation probe (bf16 + fp8 arms), then weight-only fp8 arm. Serialized on the device.
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
EVID=$PWD/artifacts/verification-2026-10-01/ptq-wan21/phase0
mkdir -p "$EVID"
P=/home/ubuntu/.claude/jobs/b5f130d0/tmp/ptq_probe
echo "=== probe dynamic $(date -u +%FT%TZ)"
python scripts/ptq_fp8_device_probe.py --work-dir "$P" --force-clean 2>&1 | tee "$EVID/probe.log" | grep -vE 'Warning|warnings.warn' | tail -25
rc1=${PIPESTATUS[0]}
cp "$P/ptq_probe_report.json" "$EVID/ptq_probe_report.json" 2>/dev/null
echo "PROBE_DYN_RC=$rc1"
P2=/home/ubuntu/.claude/jobs/b5f130d0/tmp/ptq_probe_wo
echo "=== probe weight-only $(date -u +%FT%TZ)"
python scripts/ptq_fp8_device_probe.py --work-dir "$P2" --force-clean --quant-act none --only fp8 2>&1 | tee "$EVID/probe_weight_only.log" | grep -vE 'Warning|warnings.warn' | tail -25
rc2=${PIPESTATUS[0]}
cp "$P2/ptq_probe_report.json" "$EVID/ptq_probe_report_weight_only.json" 2>/dev/null
echo "PROBE_WO_RC=$rc2"
echo "PROBE_BOTH_DONE dyn=$rc1 wo=$rc2"
