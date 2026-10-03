#!/usr/bin/env bash
# Is the static path's clamp needed, and what does it cost?
#  1. tiny probe, static scales, gain 1            -> device static == CPU static reference?
#  2. tiny probe, static, gain 8 (past calibration) -> both clamp: still equal?
#  3. same with DIFFLET_FP8_STATIC_NO_CLAMP=1       -> device fp8 cast of >240 values: saturate (match) or NaN?
#  4. Wan 2.1 production static arm without the clamp -> DiT step vs 586.9 ms, quality vs bf16
set -uo pipefail
T=/home/ubuntu/.claude/jobs/b5f130d0/tmp
W=/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
cd "$W"
source .venv/bin/activate
export PYTHONPATH=$PWD
E=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static/clamp
mkdir -p "$E"
bash scripts/gate_idle.sh | tee "$E/gate.txt" | tail -1 | grep -q IDLE || { echo "host busy"; exit 2; }
probe() {  # probe <label> <args...>
  local label=$1; shift
  echo "=== probe $label $(date -u +%FT%TZ)"
  python scripts/ptq_fp8_device_probe.py --work-dir "$T/ptq_probe_$label" --force-clean --static "$@" 2>&1 \
    | grep -vE 'Warning|warnings.warn|import_nki' | tee "$E/probe_$label.log" | grep -E 'calibrated|cosine|snr_db|nonfinite|absmax|passed|rror' | head -20
  echo "PROBE_${label}_RC=${PIPESTATUS[0]}"
  cp "$T/ptq_probe_$label/ptq_probe_report.json" "$E/report_$label.json" 2>/dev/null
}
probe static_g1
probe static_g8_clamp --input-gain 8
DIFFLET_FP8_STATIC_NO_CLAMP=1 probe static_g8_noclamp --input-gain 8
CAL=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static/act_calibration_wan21.json
CACHE=/home/ubuntu/.cache/difflet-noclamp
mkdir -p "$CACHE"
for d in quantized _shared_weights; do [ -e "$CACHE/$d" ] || ln -s /home/ubuntu/.cache/difflet/$d "$CACHE/$d"; done
echo "=== Wan 2.1 static arm, no clamp $(date -u +%FT%TZ)"
DIFFLET_FP8_STATIC_NO_CLAMP=1 python scripts/ptq_fp8_ab.py --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 --seed 42 \
  --runs 1 --host-vae --only fp8 --quant-calibration "$CAL" --cache-dir "$CACHE" --out-dir "$E/ab" 2>&1 | grep -vE 'Warning|warn|import_nki' | tee "$E/ab.log" | tail -8
echo "AB_NOCLAMP_RC=${PIPESTATUS[0]}"
grep -h 'dit-step ms' "$E"/ab/logs/generate_*.log 2>/dev/null | sed 's|^|[noclamp] |' | cut -c1-160
REF=$W/artifacts/verification-2026-10-01/ptq-wan21/ab/bf16_run1_hostvae.mp4
if [ -f "$E/ab/fp8_run0.mp4" ]; then
  python scripts/ptq_compare_outputs.py --reference "$REF" --test "$E/ab/fp8_run0.mp4" --out "$E/compare_noclamp_vs_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -3
  python "$T/latent_snr.py" "$W/artifacts/verification-2026-10-01/ptq-wan21/ab/work_bf16_run1/latents.pt" "$E"/ab/work_fp8_run0/latents.pt 2>&1 | grep -v Warning
fi
echo "CLAMP_DONE"
