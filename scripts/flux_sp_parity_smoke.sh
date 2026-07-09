#!/usr/bin/env bash
# On-device Flux Megatron-SP vs dense-TP numerical parity gate (tp=2, no CP).
#
# Runs the Flux DiT backbone twice on a fixed-seed input — once dense (plain TP),
# once with Megatron-style sequence parallelism — each in its OWN process (NxD
# parallel_state initializes once per process), then compares the two output
# tensors. SP is mathematically lossless vs dense TP, so cosine should be ~1.0.
#
# Uses a reduced layer count + small resolution for a fast compile; weights load
# strict=False, so both modes use identical (subset) checkpoint weights.
#
#   bash scripts/flux_sp_parity_smoke.sh
#
# Tunables (env):
#   DIFFLET_FLUX_MODEL       (black-forest-labs/FLUX.1-dev)
#   DIFFLET_FLUX_TP_DEGREE   (2)
#   DIFFLET_FLUX_LAYERS      (2)
#   DIFFLET_FLUX_SINGLE_LAYERS (2)
#   DIFFLET_FLUX_HEIGHT      (256)
#   DIFFLET_FLUX_WIDTH       (256)
#   DIFFLET_FLUX_TEXT_SEQ_LEN (512)
#   DIFFLET_FLUX_SP_PARITY_COSINE_MIN (0.999)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"
[[ -d "${NEURON_VENV}/bin" ]] && export PATH="${NEURON_VENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

TP_DEGREE="${DIFFLET_FLUX_TP_DEGREE:-2}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-${TP_DEGREE}}"
COSINE_MIN="${DIFFLET_FLUX_SP_PARITY_COSINE_MIN:-0.999}"
WORKDIR="${DIFFLET_FLUX_SP_PARITY_WORKDIR:-/tmp/flux_sp_parity}"
mkdir -p "${WORKDIR}"

cd "${ROOT}"

echo "[sp-parity] === dense ==="
"${PYTHON_BIN}" scripts/flux_sp_parity_smoke.py --mode dense --out "${WORKDIR}/out_dense.pt"

echo "[sp-parity] === sp ==="
"${PYTHON_BIN}" scripts/flux_sp_parity_smoke.py --mode sp --out "${WORKDIR}/out_sp.pt"

echo "[sp-parity] === compare ==="
COSINE_MIN="${COSINE_MIN}" WORKDIR="${WORKDIR}" "${PYTHON_BIN}" - <<'PY'
import os
import torch
import torch.nn.functional as F

wd = os.environ["WORKDIR"]
cmin = float(os.environ["COSINE_MIN"])
a = torch.load(os.path.join(wd, "out_dense.pt")).float()
b = torch.load(os.path.join(wd, "out_sp.pt")).float()
if tuple(a.shape) != tuple(b.shape):
    raise AssertionError(f"shape mismatch: dense={tuple(a.shape)} sp={tuple(b.shape)}")
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
print("[sp-parity] metrics:", metrics)
assert cos >= cmin, f"sp vs dense cosine {cos:.6f} < {cmin} (NOT lossless): {metrics}"
print(f"[sp-parity] PASS — SP matches dense TP (cosine {cos:.6f} >= {cmin})")
PY
