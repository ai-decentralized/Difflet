#!/usr/bin/env bash
# Wan 2.1 device probe with the REAL weights (first N blocks), real UMT5 text, production shape:
# device fp8-dynamic vs CPU fp8-dynamic with the real activation statistics.
set -uo pipefail
T=/home/ubuntu/.claude/jobs/b5f130d0/tmp
until grep -qs 'POW2_DONE' "$T/logs/pow2.log" && [ -f "$T/wan_text.pt" ]; do sleep 30; done
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
E=$PWD/artifacts/verification-2026-10-03/fp8-dyn-step/static/real_probe
SNAP=/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer
mkdir -p "$E"
run() {  # run <label> <num_layers> [extra args]
  local label=$1 nl=$2; shift 2
  echo "=== wan real probe $label ($nl blocks) $(date -u +%FT%TZ)"
  python scripts/ptq_fp8_device_probe.py --work-dir "$T/wan_real_$label" --force-clean \
    --real-model-dir "$SNAP" --text-pt "$T/wan_text.pt" --text-seq-len 512 \
    --height 480 --width 832 --num-frames 9 --num-layers "$nl" --iters 3 "$@" 2>&1 \
    | grep -vE 'Warning|warnings.warn|import_nki' | tee "$E/probe_$label.log" | grep -E 'real weights|cosine|snr_db|nonfinite|absmax|passed|rror' | head -24
  echo "PROBE_${label}_RC=${PIPESTATUS[0]}"
  cp "$T/wan_real_$label/ptq_probe_report.json" "$E/report_$label.json" 2>/dev/null
}
run b2 2
run b6 6
echo "WAN_REAL_PROBES_DONE"
