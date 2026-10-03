#!/usr/bin/env bash
# Tiny Wan device probe with heavy-tailed inputs: does device fp8-dynamic still match the CPU
# fp8-dynamic reference when the activations carry outliers (as the real model's do)?
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
T=/home/ubuntu/.claude/jobs/b5f130d0/tmp
E=$PWD/artifacts/verification-2026-10-03/fp8-dyn-step/static/outlier_probe
mkdir -p "$E"
run() {  # run <label> <args...>
  local label=$1; shift
  echo "=== probe $label $(date -u +%FT%TZ)"
  python scripts/ptq_fp8_device_probe.py --work-dir "$T/ptq_probe_$label" --force-clean "$@" 2>&1 \
    | grep -vE 'Warning|warnings.warn|import_nki' | tee "$E/probe_$label.log" | grep -E 'cosine|snr_db|passed|PROBE|rror' | head -20
  echo "PROBE_${label}_RC=${PIPESTATUS[0]}"
  cp "$T/ptq_probe_$label/ptq_probe_report.json" "$E/report_$label.json" 2>/dev/null
}
run normal
run out1e3x50  --outlier-frac 0.001 --outlier-mag 50
run out1e2x200 --outlier-frac 0.01 --outlier-mag 200
echo "OUTLIER_PROBES_DONE"
