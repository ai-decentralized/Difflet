#!/usr/bin/env bash
# One denoise step at full depth on the device: static arm (default cache, compiled) and the
# forced-dynamic arm (difflet-forcedyn cache, compiled), same seed; latents compared to each other
# and to the CPU bf16 one-step reference.
set -uo pipefail
T=/home/ubuntu/.claude/jobs/b5f130d0/tmp
W=/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
until grep -qs 'WAN_TP4B_DONE' "$T/logs/wan_real_tp4b.log"; do sleep 30; done
cd "$W"
source .venv/bin/activate
export PYTHONPATH=$PWD
E=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static/onestep
CAL=$W/artifacts/verification-2026-10-03/fp8-dyn-step/static/act_calibration_wan21.json
mkdir -p "$E"
bash scripts/gate_idle.sh | tee "$E/gate.txt" | tail -1 | grep -q IDLE || { echo "host busy"; exit 2; }
COMMON="--model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 --tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 1 --guidance-scale 1.0 --seed 42 --runs 1 --host-vae --only fp8 --skip-quantize --skip-compile --quant-calibration $CAL"
echo "=== static, 1 step $(date -u +%FT%TZ)"
python scripts/ptq_fp8_ab.py $COMMON --out-dir "$E/static" 2>&1 | grep -vE 'Warning|warn|import_nki' | tail -4
echo "=== forced dynamic, 1 step $(date -u +%FT%TZ)"
DIFFLET_FP8_IGNORE_INPUT_SCALE=1 python scripts/ptq_fp8_ab.py $COMMON --cache-dir /home/ubuntu/.cache/difflet-forcedyn --out-dir "$E/dynamic" 2>&1 | grep -vE 'Warning|warn|import_nki' | tail -4
if [ -f "$E/cpu_bf16_1step_latents.pt" ]; then
  echo "=== latents vs CPU bf16 (1 step)"
  python "$T/latent_snr.py" "$E/cpu_bf16_1step_latents.pt" "$E"/static/work_fp8_run0/latents.pt "$E"/dynamic/work_fp8_run0/latents.pt 2>&1 | grep -v Warning
fi
echo "=== dynamic vs static (1 step)"
python "$T/latent_snr.py" "$E"/static/work_fp8_run0/latents.pt "$E"/dynamic/work_fp8_run0/latents.pt 2>&1 | grep -v Warning
echo "ONESTEP_DEV_DONE"
