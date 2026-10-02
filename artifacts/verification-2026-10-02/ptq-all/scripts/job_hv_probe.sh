#!/usr/bin/env bash
# Tiny HunyuanVideo FP8 device probes, after the Wan 2.2 run releases the device.
until [ -f ${PTQ_JOBS:-/home/ubuntu/ptq-jobs}/logs/verify_wan22.done ]; do sleep 30; done
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
T=${PTQ_JOBS:-/home/ubuntu/ptq-jobs}
E=artifacts/verification-2026-10-02/ptq-all/hunyuan_video/device_probe
mkdir -p "$E"
run() {  # run <label> <probe args...>
  local label=$1; shift
  echo "=== probe $label $(date -u +%FT%TZ)"
  python scripts/ptq_fp8_hv_device_probe.py --work-dir "$T/hv_probe_$label" --force-clean "$@" 2>&1 \
    | grep -vE 'Warning|warnings.warn|import_nki' | tee "$E/probe_$label.log" | tail -25
  echo "PROBE_${label}_RC=${PIPESTATUS[0]}"
  cp "$T/hv_probe_$label/ptq_hv_probe_report.json" "$E/report_$label.json" 2>/dev/null
}
run dyn_rawpads  --quant-act dynamic --pad-value 1.0
run dyn_zeropads --quant-act dynamic --pad-value 0.0
run wo_rawpads   --quant-act none    --pad-value 1.0
run dyn_bigger   --quant-act dynamic --pad-value 1.0 --num-single-layers 4 --text-seq-len 256 --valid-text-rows 13 --height 128 --width 128 --num-frames 9
echo "PROBES_DONE"
