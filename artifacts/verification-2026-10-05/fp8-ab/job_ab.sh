#!/usr/bin/env bash
# Full Wan 2.1 20-step A/B of the fp8 step-time levers, 2026-10-05 (trn2.3xlarge, tp4, 480x832x9,
# seed 42, host VAE, 2 generate runs per arm: run 0 cold, run 1 warm).
#
# Waits for the 2-block screen queue (job_screen.sh) to leave the device, runs one extra screen
# config (padding without attention bounds, timing only), then three A/B configurations, each in
# its own compile cache (the stage key does not see the env switches):
#   levers_bf16q : VC2 + strided DMA, fp8 also with bf16-domain quantize   (bf16 control + fp8)
#   levers       : VC2 + strided DMA, plain fp8 static                      (fp8 only)
#   shipped      : no switches, this host's reference                       (bf16 + fp8)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PATH="/home/ubuntu/Difflet/.venv/bin:${PATH}"
export PYTHONPATH="${ROOT}"
E="${ROOT}/artifacts/verification-2026-10-05/fp8-ab"
CAL="${ROOT}/artifacts/verification-2026-10-03/fp8-dyn-step/static/act_calibration_wan21.json"
REV=38ec498cb3208fb688890f8cc7e94ede2cbd7f68
mkdir -p "${E}"
exec >> "${E}/job_ab.out" 2>&1

while pgrep -f job_screen.sh >/dev/null; do sleep 30; done
# (first launch 16:48: pad128_nobounds screen ran here; then the A/B failed at compile because the
#  re-downloaded HF cache had no refs/main and the stage resolves the model offline without a
#  revision -> refs/main written to the pinned snapshot; relaunched 17:1x.)

ab() {  # ab <name> <only> [KEY=VALUE ...]
  local name="$1" only="$2"; shift 2
  echo "=== $(date -u +%T) A/B ${name} (--only ${only}) switches: $*"
  ( for kv in "$@"; do export "$kv"; done
    python scripts/ptq_fp8_ab.py --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision "${REV}" \
      --tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 --seed 42 \
      --runs 2 --host-vae --only "${only}" --quant-calibration "${CAL}" \
      --cache-dir "/home/ubuntu/.cache/difflet-ab-${name}" --out-dir "${E}/${name}" \
      > "${E}/${name}.log" 2>&1 )
  echo "=== $(date -u +%T) A/B ${name} exit $?"
  grep -h 'dit-step ms\|dit_step' "${E}/${name}/ab_summary.md" 2>/dev/null | head -6
}

ab levers_bf16q both NEURON_RT_VIRTUAL_CORE_SIZE=2 DIFFLET_WAN_TENSORIZER_EXTRA=--vectorize-strided-dma DIFFLET_FP8_BF16_QUANT=1
ab levers       fp8  NEURON_RT_VIRTUAL_CORE_SIZE=2 DIFFLET_WAN_TENSORIZER_EXTRA=--vectorize-strided-dma
ab shipped      both

echo "=== $(date -u +%T) cross-config quality: fp8 renders vs the levers bf16 control"
REF="${E}/levers_bf16q/bf16_run1.mp4"
for n in levers_bf16q levers shipped; do
  [ -f "${E}/${n}/fp8_run1.mp4" ] || continue
  python scripts/ptq_compare_outputs.py --reference "${REF}" --test "${E}/${n}/fp8_run1.mp4" \
    --latents-reference "${E}/levers_bf16q/work_bf16_run1/latents.pt" --latents-test "${E}/${n}/work_fp8_run1/latents.pt" \
    --out "${E}/compare_${n}_fp8_vs_levers_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -2
done
[ -f "${E}/shipped/bf16_run1.mp4" ] && python scripts/ptq_compare_outputs.py --reference "${REF}" --test "${E}/shipped/bf16_run1.mp4" \
    --out "${E}/compare_shipped_bf16_vs_levers_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -1
echo "=== $(date -u +%T) AB_DONE"
