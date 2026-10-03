#!/usr/bin/env bash
# Wan 2.1 probes under TP4 (the production topology no probe has exercised): the random tiny model
# with outliers and the real 2-block model with real text, device fp8-dynamic vs CPU fp8-dynamic.
set -uo pipefail
T=/home/ubuntu/.claude/jobs/b5f130d0/tmp
until grep -qs 'WAN_REAL_PROBES_DONE' "$T/logs/wan_real_probe.log"; do sleep 30; done
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
E=$PWD/artifacts/verification-2026-10-03/fp8-dyn-step/static/real_probe
SNAP=/home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer
mkdir -p "$E"
run() {  # run <label> <args...>
  local label=$1; shift
  echo "=== wan tp4 probe $label $(date -u +%FT%TZ)"
  python scripts/ptq_fp8_device_probe.py --work-dir "$T/wan_tp4_$label" --force-clean --tp-degree 4 --iters 3 "$@" 2>&1 \
    | grep -vE 'Warning|warnings.warn|import_nki' | tee "$E/probe_tp4_$label.log" | grep -E 'real weights|cosine|snr_db|nonfinite|absmax|passed|rror' | head -24
  echo "PROBE_tp4_${label}_RC=${PIPESTATUS[0]}"
  cp "$T/wan_tp4_$label/ptq_probe_report.json" "$E/report_tp4_$label.json" 2>/dev/null
}
run tiny_out --outlier-frac 0.01 --outlier-mag 200
run real_b2 --real-model-dir "$SNAP" --text-pt "$T/wan_text.pt" --text-seq-len 512 --height 480 --width 832 --num-frames 9 --num-layers 2
echo "WAN_TP4_PROBES_DONE"
