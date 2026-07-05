#!/usr/bin/env bash
# On-device parallel-mesh bit-identity regression + smoke gate, ALL models.
#
# For each (model, mode) pair covering every parallel axis the mesh refactor
# touched — wan:cfg wan:cp flux:cp hunyuan:cp qwen:cp ltx2:cfg — runs the tiny
# seeded DiT backbone twice on a fixed-seed input: once with difflet from the
# PRE-refactor baseline commit (git worktree), once from this checkout, then
# byte-compares the outputs. The refactor only changes WHICH subgroup each
# collective fires in (identical replica groups for these configs), so the
# outputs must be bit-identical: PSNR = inf.
#
# HunyuanVideo-1.5 has no CFG/CP wiring (both rejected at entry), so it gets a
# plain tp-only compile/forward smoke (hunyuan15_tiny_compile_smoke.py) on the
# refactor checkout only.
#
#   bash scripts/mesh_regression_smoke.sh
#
# Tunables (env): DIFFLET_MESH_BASELINE_REF (93fac4c), DIFFLET_MESH_WORKDIR,
#   DIFFLET_MESH_PAIRS ("wan:cfg wan:cp flux:cp hunyuan:cp qwen:cp ltx2:cfg"),
#   DIFFLET_MESH_SKIP_HYV15 (0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"
[[ -d "${NEURON_VENV}/bin" ]] && export PATH="${NEURON_VENV}/bin:${PATH}"
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-4}"

BASE_REF="${DIFFLET_MESH_BASELINE_REF:-93fac4c}"
WORKDIR="${DIFFLET_MESH_WORKDIR:-/tmp/mesh_regression}"
PAIRS="${DIFFLET_MESH_PAIRS:-wan:cfg wan:cp flux:cp hunyuan:cp qwen:cp ltx2:cfg}"
BASELINE_DIR="${WORKDIR}/baseline_checkout"
RUNNER="${ROOT}/scripts/mesh_regression_smoke.py"
mkdir -p "${WORKDIR}"

# 1. Baseline checkout (detached worktree at the pre-refactor commit).
if [[ ! -d "${BASELINE_DIR}" ]]; then
  git -C "${ROOT}" worktree add --detach "${BASELINE_DIR}" "${BASE_REF}"
fi
echo "[mesh-regression] baseline = $(git -C "${BASELINE_DIR}" rev-parse --short HEAD)"

cd "${WORKDIR}"

# 2. Per-pair: shared tiny seeded checkpoint, then baseline + refactor runs.
for pair in ${PAIRS}; do
  model="${pair%%:*}"
  mode="${pair##*:}"
  CKPT_DIR="${WORKDIR}/tiny_${model}_ckpt"
  if [[ ! -f "${CKPT_DIR}/transformer/config.json" ]]; then
    PYTHONPATH="${ROOT}" "${PYTHON_BIN}" "${RUNNER}" --model "${model}" --make-checkpoint "${CKPT_DIR}"
  fi

  # Resumable: skip a run whose output already exists (delete the .pt to redo).
  if [[ ! -f "${WORKDIR}/baseline_${model}_${mode}.pt" ]]; then
    echo "[mesh-regression] === ${model}:${mode} baseline (${BASE_REF}) ==="
    PYTHONPATH="${BASELINE_DIR}" "${PYTHON_BIN}" "${RUNNER}" \
      --model "${model}" --mode "${mode}" --ckpt "${CKPT_DIR}" \
      --out "${WORKDIR}/baseline_${model}_${mode}.pt" \
      --work-dir "${WORKDIR}/baseline_${model}_${mode}_work"
  fi

  if [[ ! -f "${WORKDIR}/refactor_${model}_${mode}.pt" ]]; then
    echo "[mesh-regression] === ${model}:${mode} refactor (HEAD) ==="
    PYTHONPATH="${ROOT}" "${PYTHON_BIN}" "${RUNNER}" \
      --model "${model}" --mode "${mode}" --ckpt "${CKPT_DIR}" \
      --out "${WORKDIR}/refactor_${model}_${mode}.pt" \
      --work-dir "${WORKDIR}/refactor_${model}_${mode}_work"
  fi
done

# 3. Byte-compare every pair.
WORKDIR="${WORKDIR}" PAIRS="${PAIRS}" "${PYTHON_BIN}" - <<'PY'
import math
import os
import torch

wd = os.environ["WORKDIR"]
failed = False


def compare(name, a, b):
    global failed
    identical = a.shape == b.shape and torch.equal(a, b)
    if identical:
        psnr = "inf"
    else:
        mse = torch.mean((a.float() - b.float()) ** 2).item()
        peak = a.float().abs().max().item()
        psnr = "inf" if mse == 0 else f"{10 * math.log10((peak ** 2) / mse):.2f}"
        failed = True
    print(f"[mesh-regression] {name} shape={tuple(a.shape)} bit_identical={identical} PSNR={psnr}")


for pair in os.environ["PAIRS"].split():
    model, mode = pair.split(":")
    a = torch.load(os.path.join(wd, f"baseline_{model}_{mode}.pt"))
    b = torch.load(os.path.join(wd, f"refactor_{model}_{mode}.pt"))
    if isinstance(a, dict):
        for key in a:
            compare(f"{model}:{mode}:{key}", a[key], b[key])
    else:
        compare(f"{model}:{mode}", a, b)
raise SystemExit(1 if failed else 0)
PY
echo "[mesh-regression] PASS: all model/mode pairs bit-identical to ${BASE_REF}"

# 4. HunyuanVideo-1.5 tp-only compile/forward smoke (no parallel-axis change).
if [[ "${DIFFLET_MESH_SKIP_HYV15:-0}" != "1" ]]; then
  echo "[mesh-regression] === hyv15 tp-only smoke (refactor) ==="
  PYTHONPATH="${ROOT}" "${PYTHON_BIN}" "${ROOT}/scripts/hunyuan15_tiny_compile_smoke.py" \
    --model-dir "${WORKDIR}/tiny_hyv15_model" --cache-dir "${WORKDIR}/tiny_hyv15_cache" \
    --tp-degree 4 --load
  echo "[mesh-regression] PASS: hyv15 smoke"
fi
echo "[mesh-regression] ALL PASS"
