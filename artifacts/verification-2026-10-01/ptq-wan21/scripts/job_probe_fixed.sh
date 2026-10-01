#!/usr/bin/env bash
# Tiny probe with the fixed layers (42ae642): bf16 + fp8-dyn, then fp8 weight-only. Evidence under phase0-fixed/.
set -uo pipefail
until [ -f /home/ubuntu/.claude/jobs/b5f130d0/tmp/logs/chain2.done ]; do sleep 30; done
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
EVID=$PWD/artifacts/verification-2026-10-01/ptq-wan21/phase0-fixed
mkdir -p "$EVID"
bash /home/ubuntu/.claude/jobs/b5f130d0/tmp/gate_idle.sh | tee "$EVID/gate.txt" | tail -1
P=/home/ubuntu/.claude/jobs/b5f130d0/tmp/ptq_probe_fixed
echo "=== probe dynamic (fixed layers) $(date -u +%FT%TZ)"
python scripts/ptq_fp8_device_probe.py --work-dir "$P" --force-clean 2>&1 | tee "$EVID/probe.log" | grep -vE 'Warning|warnings.warn' | tail -12
echo "PROBE_DYN_RC=${PIPESTATUS[0]}"
cp "$P/ptq_probe_report.json" "$EVID/ptq_probe_report.json" 2>/dev/null
P2=/home/ubuntu/.claude/jobs/b5f130d0/tmp/ptq_probe_fixed_wo
echo "=== probe weight-only (fixed layers) $(date -u +%FT%TZ)"
python scripts/ptq_fp8_device_probe.py --work-dir "$P2" --force-clean --quant-act none --only fp8 2>&1 | tee "$EVID/probe_weight_only.log" | grep -vE 'Warning|warnings.warn' | tail -12
echo "PROBE_WO_RC=${PIPESTATUS[0]}"
cp "$P2/ptq_probe_report.json" "$EVID/ptq_probe_report_weight_only.json" 2>/dev/null
echo "PROBE_FIXED_DONE"
