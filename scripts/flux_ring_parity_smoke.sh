#!/usr/bin/env bash
# On-device Flux ring-vs-gather-KV numerical parity gate (tp=2/cp=2, LNC2).
#
# Runs the Flux DiT backbone twice on a fixed-seed input — once with
# cp_mode=gather_kv, once with cp_mode=ring — each in its OWN process (NxD
# parallel_state initializes once per process), then compares the two output
# tensors. Ring is numerically lossless vs gather-KV, so cosine should be ~1.0.
#
# Uses reduced layer count + small resolution for a fast compile; weights load
# strict=False, so only the first blocks use real checkpoint weights
# (identical across both modes → a fair compare).
#
#   bash scripts/flux_ring_parity_smoke.sh
#
# Tunables (env):
#   DIFFLET_FLUX_MODEL       (black-forest-labs/FLUX.1-dev)
#   DIFFLET_FLUX_TP_DEGREE   (2)
#   DIFFLET_FLUX_CP_DEGREE   (2)
#   DIFFLET_FLUX_LAYERS      (2)
#   DIFFLET_FLUX_SINGLE_LAYERS (2)
#   DIFFLET_FLUX_HEIGHT      (256)
#   DIFFLET_FLUX_WIDTH       (256)
#   DIFFLET_FLUX_TEXT_SEQ_LEN (512)
#   DIFFLET_FLUX_PARITY_COSINE_MIN (0.999)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"
[[ -d "${NEURON_VENV}/bin" ]] && export PATH="${NEURON_VENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

TP_DEGREE="${DIFFLET_FLUX_TP_DEGREE:-2}"
CP_DEGREE="${DIFFLET_FLUX_CP_DEGREE:-2}"
WORLD_SIZE=$(( TP_DEGREE * CP_DEGREE ))
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-${WORLD_SIZE}}"
COSINE_MIN="${DIFFLET_FLUX_PARITY_COSINE_MIN:-0.999}"
WORKDIR="${DIFFLET_FLUX_PARITY_WORKDIR:-/tmp/flux_ring_parity}"
mkdir -p "${WORKDIR}"

cd "${ROOT}"

echo "[parity] === gather_kv ==="
"${PYTHON_BIN}" scripts/flux_ring_parity_smoke.py --mode gather_kv --out "${WORKDIR}/out_gather_kv.pt"

echo "[parity] === ring ==="
"${PYTHON_BIN}" scripts/flux_ring_parity_smoke.py --mode ring --out "${WORKDIR}/out_ring.pt"

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
