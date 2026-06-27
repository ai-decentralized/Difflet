#!/usr/bin/env bash
# On-device HunyuanVideo ring-vs-gather-KV e2e trajectory parity gate (tp=2/cp=2).
#
# Runs the HunyuanVideo DiT backbone through a short denoising loop (default 3
# steps) twice — once with cp_mode=gather_kv, once with cp_mode=ring — each in
# its OWN process (NxD parallel_state initializes once per process), then
# compares the per-step latent trajectory step-by-step. Ring is numerically
# lossless vs gather-KV so every per-step cosine should be ~1.0.
#
# Uses a reduced layer count + small resolution for a fast compile; weights load
# strict=False, so both modes use identical (subset) checkpoint weights.
# Inputs are synthetic (fixed-seed random tensors): no text encoder or VAE needed.
#
#   bash scripts/hunyuan_ring_e2e_parity_smoke.sh
#
# Tunables (env):
#   DIFFLET_HUNYUAN_MODEL             (hunyuanvideo-community/HunyuanVideo)
#   DIFFLET_HUNYUAN_TP_DEGREE         (2)
#   DIFFLET_HUNYUAN_CP_DEGREE         (2)
#   DIFFLET_HUNYUAN_LAYERS            (2)
#   DIFFLET_HUNYUAN_HEIGHT            (256)
#   DIFFLET_HUNYUAN_WIDTH             (256)
#   DIFFLET_HUNYUAN_NUM_FRAMES        (5)
#   DIFFLET_HUNYUAN_E2E_STEPS         (3)
#   DIFFLET_HUNYUAN_E2E_COSINE_MIN    (0.999)
#
# NOTE: This script is committed unrun. HunyuanVideo transformer weights are not
# cached on the compile box and the disk cannot fit them. Run on a box with the
# real weights and Trainium2 hardware (set DIFFLET_RUN_HUNYUAN_RING_E2E=1).
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
COSINE_MIN="${DIFFLET_HUNYUAN_E2E_COSINE_MIN:-0.999}"
WORKDIR="${DIFFLET_HUNYUAN_E2E_WORKDIR:-/tmp/hunyuan_ring_e2e_parity}"
mkdir -p "${WORKDIR}"

cd "${ROOT}"

echo "[e2e-parity] === gather_kv ==="
"${PYTHON_BIN}" scripts/hunyuan_ring_e2e_parity_smoke.py \
    --mode gather_kv \
    --out "${WORKDIR}/traj_gather_kv.pt"

echo "[e2e-parity] === ring ==="
"${PYTHON_BIN}" scripts/hunyuan_ring_e2e_parity_smoke.py \
    --mode ring \
    --out "${WORKDIR}/traj_ring.pt"

echo "[e2e-parity] === compare ==="
COSINE_MIN="${COSINE_MIN}" WORKDIR="${WORKDIR}" "${PYTHON_BIN}" - <<'PY'
import os
import torch
import torch.nn.functional as F

wd = os.environ["WORKDIR"]
cmin = float(os.environ["COSINE_MIN"])
traj_gkv = torch.load(os.path.join(wd, "traj_gather_kv.pt"))
traj_ring = torch.load(os.path.join(wd, "traj_ring.pt"))

if len(traj_gkv) != len(traj_ring):
    raise AssertionError(
        f"trajectory length mismatch: gather_kv={len(traj_gkv)} ring={len(traj_ring)}"
    )

cosines = []
for i, (a, b) in enumerate(zip(traj_gkv, traj_ring)):
    a, b = a.float(), b.float()
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(
            f"step {i} shape mismatch: gather_kv={tuple(a.shape)} ring={tuple(b.shape)}"
        )
    diff = (a - b).abs()
    cos = float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    cosines.append(cos)
    metrics = {
        "step": i,
        "cosine": cos,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(torch.sqrt((diff * diff).mean())),
    }
    print(f"[e2e-parity] step {i} metrics: {metrics}")

min_cos = min(cosines)
print(f"[e2e-parity] MIN cosine over {len(cosines)} steps: {min_cos:.6f} (threshold={cmin})")
failures = [(i, c) for i, c in enumerate(cosines) if c < cmin]
if failures:
    raise AssertionError(
        f"ring vs gather_kv trajectory parity FAILED at steps {failures}: "
        f"min_cosine={min_cos:.6f} < {cmin}"
    )
print(f"[e2e-parity] PASS — ring matches gather_kv e2e trajectory "
      f"(min cosine {min_cos:.6f} >= {cmin} over {len(cosines)} steps)")
PY
