#!/usr/bin/env bash
# Wan 2.1 production path with the new defaults (VC2 for the transformer stage, strided DMA in
# the backbone's compiler args): no environment switches, fresh compile cache. Expected from the
# levers A/B: bf16 ~486 ms, fp8 static ~473 ms per DiT step, same renders.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PATH="/home/ubuntu/Difflet/.venv/bin:${PATH}"
export PYTHONPATH="${ROOT}"
E="${ROOT}/artifacts/verification-2026-10-05/fp8-ab"
CAL="${ROOT}/artifacts/verification-2026-10-03/fp8-dyn-step/static/act_calibration_wan21.json"
exec >> "${E}/job_defaults.out" 2>&1
while pgrep -f 'run_all.sh|job_screen2.sh|ptq_fp8_device_probe.py' >/dev/null; do sleep 30; done
rm -rf /tmp/nxd_model
echo "=== $(date -u +%T) A/B defaults (--only both), no switches"
python scripts/ptq_fp8_ab.py --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 --seed 42 \
  --runs 2 --host-vae --only both --quant-calibration "${CAL}" \
  --cache-dir /home/ubuntu/.cache/difflet-ab-defaults --out-dir "${E}/defaults" > "${E}/defaults.log" 2>&1
echo "=== $(date -u +%T) A/B defaults exit $?"
python scripts/ptq_compare_outputs.py --reference "${E}/levers_bf16q/bf16_run1.mp4" --test "${E}/defaults/bf16_run1.mp4" \
  --out "${E}/compare_defaults_bf16_vs_levers_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -1
echo "=== $(date -u +%T) DEFAULTS_DONE"
