#!/usr/bin/env bash
# cclog 89: Wan real-prompt fixed-cadence TeaCache A/B (device, stage-1 denoise only).
# Runs the 50-step denoise twice with the SAME seeded init noise + real UMT5 prompt:
#   (1) baseline  — no teacache, 50 full steps
#   (2) teacache  — fixed-cadence calib (skip every 2nd step in [2,48))
# then reports final-latent cosine + denoise speedup ([wan] forward elapsed).
# VAE is skipped (--no-vae): the open question is denoise quality, decode is identical.
set -uo pipefail
ROOT=/home/ubuntu/nova
NEURON_VENV=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
export PATH="${NEURON_VENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}"
export NOVA_BACKEND=trainium
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_NUM_CORES=4
cd "${ROOT}"

PROMPT="${NOVA_WAN_PROMPT:-a cat walking in a garden}"
STEPS="${NOVA_WAN_STEPS:-50}"
CALIB="${ROOT}/cclogs/m9-teacache/teacache_calib_wan_cadence.json"

COMMON=(--model Wan-AI/Wan2.2-T2V-A14B-Diffusers --tp-degree 4 --skip-warmup
  --num-inference-steps "${STEPS}" --num-frames 9 --height 480 --width 832
  --prompt "${PROMPT}" --enable-text --enable-transformer --enable-transformer-2 --no-vae
  --output-type latent)

echo "===== [wan-ab] BASELINE (no teacache, ${STEPS} full steps) ====="
python examples/wan_example.py "${COMMON[@]}" \
  --save-latents "${ROOT}/.nova-cache/wan_ab_baseline.pt" \
  --compiled-dir "${ROOT}/.nova-cache/wan_ab_run_baseline"
echo "[wan-ab] baseline exit=$?"

echo "===== [wan-ab] TEACACHE (fixed-cadence) ====="
python examples/wan_example.py "${COMMON[@]}" \
  --teacache-calibration "${CALIB}" \
  --save-latents "${ROOT}/.nova-cache/wan_ab_teacache.pt" \
  --compiled-dir "${ROOT}/.nova-cache/wan_ab_run_teacache"
echo "[wan-ab] teacache exit=$?"

echo "===== [wan-ab] COMPARE ====="
python - <<'PY'
import torch, torch.nn.functional as F
b = torch.load('/home/ubuntu/nova/.nova-cache/wan_ab_baseline.pt', map_location='cpu').float().flatten()
t = torch.load('/home/ubuntu/nova/.nova-cache/wan_ab_teacache.pt', map_location='cpu').float().flatten()
cos = float(F.cosine_similarity(b, t, dim=0))
rel = float((b - t).norm() / b.norm().clamp_min(1e-8))
print(f"[wan-ab] RESULT final-latent cosine = {cos:.6f}   rel-L2 = {rel:.6f}")
PY
