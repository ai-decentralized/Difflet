#!/usr/bin/env bash
# Weight-only fp8 probe (--quant-act none): A1/A4/A5 independent of the dynamic-activation path.
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
EVID=$PWD/artifacts/verification-2026-10-01/ptq-wan21/phase0
PROBE=/home/ubuntu/.claude/jobs/b5f130d0/tmp/ptq_probe_wo
echo "=== probe weight-only $(date -u +%FT%TZ)"
python scripts/ptq_fp8_device_probe.py --work-dir "$PROBE" --force-clean --quant-act none --only fp8 2>&1 | tee "$EVID/probe_weight_only.log" | tail -30
rc=${PIPESTATUS[0]}
echo "PROBE_RC=$rc"
cp "$PROBE/ptq_probe_report.json" "$EVID/ptq_probe_report_weight_only.json" 2>/dev/null
echo "PROBE_WO_DONE rc=$rc"
