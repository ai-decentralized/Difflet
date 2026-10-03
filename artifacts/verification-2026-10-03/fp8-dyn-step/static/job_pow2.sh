#!/usr/bin/env bash
# Diagnostic: dynamic law with a power-of-two scale (DIFFLET_FP8_DYN_POW2=1) on the static
# checkpoint (input_scale ignored), Wan 2.1, separate cache dir. Tests whether a step-to-step
# coherent fp8 grid recovers the static arm's end-to-end fidelity.
set -uo pipefail
T=/home/ubuntu/.claude/jobs/b5f130d0/tmp
W=/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
until grep -qs 'REAL_PROBES_DONE' "$T/logs/hv_real_probe.log"; do sleep 30; done
cd "$W"
source .venv/bin/activate
export PYTHONPATH=$PWD
export DIFFLET_FP8_IGNORE_INPUT_SCALE=1 DIFFLET_FP8_DYN_POW2=1
E=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static/pow2_dynamic
CAL=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static/act_calibration_wan21.json
mkdir -p "$E"
bash scripts/gate_idle.sh | tee "$E/gate.txt" | tail -1 | grep -q IDLE || { echo "host busy"; exit 2; }
CACHE=/home/ubuntu/.cache/difflet-pow2
mkdir -p "$CACHE"
for d in quantized _shared_weights; do [ -e "$CACHE/$d" ] || ln -s /home/ubuntu/.cache/difflet/$d "$CACHE/$d"; done
# reuse the text-encoder stage artifacts compiled for the forced-dynamic run
for d in wan2_1_t2v_14b_diffusers_transformer wan_transformer; do :; done
echo "=== pow2-dynamic Wan 2.1 arm $(date -u +%FT%TZ)"
python scripts/ptq_fp8_ab.py --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 --seed 42 \
  --runs 1 --host-vae --only fp8 --quant-calibration "$CAL" --cache-dir "$CACHE" --out-dir "$E/ab" 2>&1 | grep -vE 'Warning|warn|import_nki' | tee "$E/ab.log" | tail -12
echo "AB_POW2_RC=${PIPESTATUS[0]}"
grep -h 'dit-step ms' "$E"/ab/logs/generate_*.log 2>/dev/null | sed 's|^|[pow2] |' | cut -c1-160
REF=$W/artifacts/verification-2026-10-01/ptq-wan21/ab/bf16_run1_hostvae.mp4
if [ -f "$E/ab/fp8_run0.mp4" ]; then
  python scripts/ptq_compare_outputs.py --reference "$REF" --test "$E/ab/fp8_run0.mp4" --out "$E/compare_pow2_vs_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -3
  python "$T/latent_snr.py" "$W/artifacts/verification-2026-10-01/ptq-wan21/ab/work_bf16_run1/latents.pt" "$E"/ab/work_fp8_run0/latents.pt 2>&1 | grep -v Warning
fi
echo "POW2_DONE"
