#!/usr/bin/env bash
# On-device Wan CP-mode-vs-gather-KV numerical parity gate (tp=2/cp=2).
#
# Runs the Wan DiT backbone twice on a fixed-seed input — once with
# cp_mode=gather_kv (the baseline), once with the candidate cp_mode — each in its
# OWN process (NxD parallel_state initializes once per process), then compares the
# two output latents. Both ring and ulysses are numerically lossless vs gather-KV,
# so cosine should be ~1.0 either way.
#
# Uses a reduced layer count + small resolution for a fast compile; weights load
# strict=False, so both modes use identical (subset) checkpoint weights.
#
#   bash scripts/wan_ring_parity_smoke.sh            # candidate: ring (default)
#   bash scripts/wan_ring_parity_smoke.sh ulysses    # candidate: ulysses
#
# Tunables (env): DIFFLET_WAN_MODEL, DIFFLET_WAN_TP_DEGREE (2), DIFFLET_WAN_CP_DEGREE (2),
#   DIFFLET_WAN_LAYERS (2), DIFFLET_WAN_HEIGHT (256), DIFFLET_WAN_WIDTH (512),
#   DIFFLET_WAN_LATENT_FRAMES (1), DIFFLET_WAN_PARITY_COSINE_MIN (0.999).
set -euo pipefail

CANDIDATE="${1:-ring}"
case "${CANDIDATE}" in
  ring|ulysses) ;;
  *) echo "usage: $0 [ring|ulysses]" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"
[[ -d "${NEURON_VENV}/bin" ]] && export PATH="${NEURON_VENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

TP_DEGREE="${DIFFLET_WAN_TP_DEGREE:-2}"
CP_DEGREE="${DIFFLET_WAN_CP_DEGREE:-2}"
WORLD_SIZE=$(( TP_DEGREE * CP_DEGREE ))
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-${WORLD_SIZE}}"
COSINE_MIN="${DIFFLET_WAN_PARITY_COSINE_MIN:-0.999}"
WORKDIR="${DIFFLET_WAN_PARITY_WORKDIR:-/tmp/wan_${CANDIDATE}_parity}"
mkdir -p "${WORKDIR}"

cd "${ROOT}"

echo "[parity] === gather_kv (baseline) ==="
"${PYTHON_BIN}" scripts/wan_ring_parity_smoke.py --mode gather_kv --out "${WORKDIR}/out_gather_kv.pt"

echo "[parity] === ${CANDIDATE} (candidate) ==="
"${PYTHON_BIN}" scripts/wan_ring_parity_smoke.py --mode "${CANDIDATE}" --out "${WORKDIR}/out_${CANDIDATE}.pt"

echo "[parity] === compare ==="
COSINE_MIN="${COSINE_MIN}" WORKDIR="${WORKDIR}" CANDIDATE="${CANDIDATE}" "${PYTHON_BIN}" - <<'PY'
import os
import torch
import torch.nn.functional as F

wd = os.environ["WORKDIR"]
cand = os.environ["CANDIDATE"]
cmin = float(os.environ["COSINE_MIN"])
a = torch.load(os.path.join(wd, "out_gather_kv.pt")).float()
b = torch.load(os.path.join(wd, f"out_{cand}.pt")).float()
if tuple(a.shape) != tuple(b.shape):
    raise AssertionError(f"shape mismatch: gather_kv={tuple(a.shape)} {cand}={tuple(b.shape)}")
diff = (a - b).abs()
cos = float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
metrics = {
    "candidate": cand,
    "shape": list(a.shape),
    "cosine": cos,
    "max_abs": float(diff.max()),
    "mean_abs": float(diff.mean()),
    "rmse": float(torch.sqrt((diff * diff).mean())),
    "cosine_min": cmin,
}
print("[parity] metrics:", metrics)
assert cos >= cmin, f"{cand} vs gather_kv cosine {cos:.6f} < {cmin} (NOT lossless): {metrics}"
print(f"[parity] PASS — {cand} matches gather_kv (cosine {cos:.6f} >= {cmin})")
PY
