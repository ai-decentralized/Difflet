#!/usr/bin/env bash
# On-device HunyuanVideo ring-vs-gather-KV numerical parity gate (tp=2/cp=2).
#
# Runs the HunyuanVideo DiT backbone twice on a fixed-seed input — once with
# cp_mode=gather_kv, once with cp_mode=ring — each in its OWN process (NxD
# parallel_state initializes once per process), then compares the two output
# latents. Ring is numerically lossless vs gather-KV, so cosine should be ~1.0.
#
# Uses a reduced layer count + small resolution for a fast compile; weights load
# strict=False, so both modes use identical (subset) checkpoint weights.
#
#   bash scripts/hunyuan_ring_parity_smoke.sh
#
# Tunables (env):
#   DIFFLET_HUNYUAN_MODEL        (hunyuanvideo-community/HunyuanVideo)
#   DIFFLET_HUNYUAN_TP_DEGREE    (2)
#   DIFFLET_HUNYUAN_CP_DEGREE    (2)
#   DIFFLET_HUNYUAN_LAYERS       (2)
#   DIFFLET_HUNYUAN_HEIGHT       (256)
#   DIFFLET_HUNYUAN_WIDTH        (256)
#   DIFFLET_HUNYUAN_NUM_FRAMES   (5)
#   DIFFLET_HUNYUAN_PARITY_COSINE_MIN  (0.999)
#
# NOTE: device parity is DEFERRED — HunyuanVideo transformer weights are not
# cached on the compile box and the disk cannot fit them. This script is committed
# unrun. Run it on a box with the real weights and Trainium2 hardware.
# Generic joint-ring correctness is already proven by Task 3 (cosine 0.999972
# on trn2 at tp=2/cp=2 for the joint_ring_attention op directly).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"
[[ -d "${NEURON_VENV}/bin" ]] && export PATH="${NEURON_VENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

TP_DEGREE="${DIFFLET_HUNYUAN_TP_DEGREE:-2}"
CP_DEGREE="${DIFFLET_HUNYUAN_CP_DEGREE:-2}"
WORLD_SIZE=$(( TP_DEGREE * CP_DEGREE ))
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-${WORLD_SIZE}}"
COSINE_MIN="${DIFFLET_HUNYUAN_PARITY_COSINE_MIN:-0.999}"
WORKDIR="${DIFFLET_HUNYUAN_PARITY_WORKDIR:-/tmp/hunyuan_ring_parity}"
mkdir -p "${WORKDIR}"

cd "${ROOT}"

echo "[parity] === gather_kv ==="
"${PYTHON_BIN}" scripts/hunyuan_ring_parity_smoke.py --mode gather_kv --out "${WORKDIR}/out_gather_kv.pt"

echo "[parity] === ring ==="
"${PYTHON_BIN}" scripts/hunyuan_ring_parity_smoke.py --mode ring --out "${WORKDIR}/out_ring.pt"

echo "[parity] === compare ==="
COSINE_MIN="${COSINE_MIN}" WORKDIR="${WORKDIR}" "${PYTHON_BIN}" - <<'PY'
import os
import torch
import torch.nn.functional as F

wd = os.environ["WORKDIR"]
cmin = float(os.environ["COSINE_MIN"])
a = torch.load(os.path.join(wd, "out_gather_kv.pt")).float()
b = torch.load(os.path.join(wd, "out_ring.pt")).float()
if tuple(a.shape) != tuple(b.shape):
    raise AssertionError(f"shape mismatch: gather_kv={tuple(a.shape)} ring={tuple(b.shape)}")
diff = (a - b).abs()
cos = float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
metrics = {
    "shape": list(a.shape),
    "cosine": cos,
    "max_abs": float(diff.max()),
    "mean_abs": float(diff.mean()),
    "rmse": float(torch.sqrt((diff * diff).mean())),
    "cosine_min": cmin,
}
print("[parity] metrics:", metrics)
assert cos >= cmin, f"ring vs gather_kv cosine {cos:.6f} < {cmin} (NOT lossless): {metrics}"
print(f"[parity] PASS — ring matches gather_kv (cosine {cos:.6f} >= {cmin})")
PY
