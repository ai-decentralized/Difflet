#!/usr/bin/env bash
# Diagnostic: the SAME static checkpoint (fp8-tensor-static-c2fa0ce2) run through the DYNAMIC
# activation law (DIFFLET_FP8_IGNORE_INPUT_SCALE=1), in a separate cache dir so nothing is reused
# except the quantized checkpoint and the shared text-encoder / VAE artifacts. If this arm renders
# at ~24.5 dB (latent SNR ~13 dB) the dynamic law is the defect; if ~33 dB the law is innocent.
set -uo pipefail
T=/home/ubuntu/.claude/jobs/b5f130d0/tmp
W=/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
until grep -qs 'CHAIN_DONE' "$T/logs/chain_static_hv.log"; do sleep 30; done
cd "$W"
source .venv/bin/activate
export PYTHONPATH=$PWD
export DIFFLET_FP8_IGNORE_INPUT_SCALE=1
E=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static/forced_dynamic
CAL=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static/act_calibration_wan21.json
mkdir -p "$E"
bash scripts/gate_idle.sh | tee "$E/gate.txt" | tail -1 | grep -q IDLE || { echo "host busy"; exit 2; }
CACHE=/home/ubuntu/.cache/difflet-forcedyn
mkdir -p "$CACHE"
# reuse the quantized checkpoints and shared weights/stage artifacts of the text encoder + VAE
for d in quantized _shared_weights; do [ -e "$CACHE/$d" ] || ln -s /home/ubuntu/.cache/difflet/$d "$CACHE/$d"; done
ls /home/ubuntu/.cache/difflet/ | head -20
echo "=== forced-dynamic Wan 2.1 arm $(date -u +%FT%TZ)"
python scripts/ptq_fp8_ab.py --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 --seed 42 \
  --runs 1 --host-vae --only fp8 --quant-calibration "$CAL" --cache-dir "$CACHE" --out-dir "$E/ab" 2>&1 | grep -vE 'Warning|warn|import_nki' | tee "$E/ab.log" | tail -12
echo "AB_FORCEDYN_RC=${PIPESTATUS[0]}"
grep -h 'dit-step ms' "$E"/ab/logs/generate_*.log 2>/dev/null | sed 's|^|[forcedyn] |' | cut -c1-160
REF=$W/artifacts/verification-2026-10-01/ptq-wan21/ab/bf16_run1_hostvae.mp4
if [ -f "$E/ab/fp8_run0.mp4" ]; then
  python scripts/ptq_compare_outputs.py --reference "$REF" --test "$E/ab/fp8_run0.mp4" --out "$E/compare_forcedyn_vs_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -4
  python "$T/latent_snr.py" "$W/artifacts/verification-2026-10-01/ptq-wan21/ab/work_bf16_run1/latents.pt" "$E"/ab/work_fp8_run0/latents.pt 2>&1 | grep -v Warning
fi
echo "FORCEDYN_DONE"
