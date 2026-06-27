#!/usr/bin/env bash
# On-device joint-MMDiT ring-vs-gather-KV numerical parity gate (Task 3, tp=2/cp=2).
#
# Compiles a tiny SPMD probe in ONE process (NxD ModelBuilder launches the whole
# tp*cp grid). Each rank scatters a synthetic image K,V across the cp ring
# (per-rank S_img/cp), keeps text K,V replicated (S_txt), forms this rank's joint
# query S_q = S_img/cp + S_txt, then computes BOTH the joint_ring_attention
# candidate (collective_permute ring) and a full gather-KV joint-attention
# reference in-graph, and compares them (cosine should be ~1.0 — lossless).
#
#   bash scripts/joint_ring_spike.sh
#
# Tunables (env): DIFFLET_JOINT_RING_TP_DEGREE (2), DIFFLET_JOINT_RING_CP_DEGREE (2),
#   DIFFLET_JOINT_RING_B (1), DIFFLET_JOINT_RING_H (8), DIFFLET_JOINT_RING_S_IMG (256),
#   DIFFLET_JOINT_RING_S_TXT (128), DIFFLET_JOINT_RING_D (128),
#   DIFFLET_JOINT_RING_COSINE_MIN (0.999).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
PYTHON_BIN="${PYTHON_BIN:-${NEURON_VENV}/bin/python}"
[[ -d "${NEURON_VENV}/bin" ]] && export PATH="${NEURON_VENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"

TP_DEGREE="${DIFFLET_JOINT_RING_TP_DEGREE:-2}"
CP_DEGREE="${DIFFLET_JOINT_RING_CP_DEGREE:-2}"
WORLD_SIZE=$(( TP_DEGREE * CP_DEGREE ))
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-${WORLD_SIZE}}"
COSINE_MIN="${DIFFLET_JOINT_RING_COSINE_MIN:-0.999}"
WORKDIR="${DIFFLET_JOINT_RING_WORKDIR:-/tmp/joint_ring_spike}"
mkdir -p "${WORKDIR}"

cd "${ROOT}"

echo "[joint-parity] === compile + run (tp=${TP_DEGREE}/cp=${CP_DEGREE}, world=${WORLD_SIZE}) ==="
DIFFLET_JOINT_RING_TP_DEGREE="${TP_DEGREE}" \
DIFFLET_JOINT_RING_CP_DEGREE="${CP_DEGREE}" \
DIFFLET_JOINT_RING_COSINE_MIN="${COSINE_MIN}" \
DIFFLET_JOINT_RING_WORKDIR="${WORKDIR}" \
"${PYTHON_BIN}" scripts/joint_ring_spike.py
