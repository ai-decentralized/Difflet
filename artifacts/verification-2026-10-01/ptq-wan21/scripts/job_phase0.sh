#!/usr/bin/env bash
# Phase 0b: quant unit tests (stage=tests) and tiny on-device FP8 probe (stage=probe). Default: both.
set -uo pipefail
STAGE="${1:-all}"
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
EVID=$PWD/artifacts/verification-2026-10-01/ptq-wan21/phase0
mkdir -p "$EVID"
if [ "$STAGE" = all ] || [ "$STAGE" = tests ]; then
  echo "=== pytest (quant) $(date -u +%FT%TZ)"
  pytest tests/unit/quant tests/unit/cli/test_cli_quant.py tests/unit/serving/test_serve_quant.py tests/unit/test_ptq_scripts.py -q -p no:cacheprovider 2>&1 | tee "$EVID/pytest_quant.log" | tail -15
  echo "PYTEST_RC=${PIPESTATUS[0]}"
fi
if [ "$STAGE" = all ] || [ "$STAGE" = probe ]; then
  echo "=== probe $(date -u +%FT%TZ)"
  PROBE=/home/ubuntu/.claude/jobs/b5f130d0/tmp/ptq_probe
  python scripts/ptq_fp8_device_probe.py --work-dir "$PROBE" --force-clean 2>&1 | tee "$EVID/probe.log" | tail -40
  rc=${PIPESTATUS[0]}
  echo "PROBE_RC=$rc"
  cp "$PROBE/ptq_probe_report.json" "$EVID/ptq_probe_report.json" 2>/dev/null
fi
echo "PHASE0_DONE stage=$STAGE"
