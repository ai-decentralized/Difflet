#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/difflet}"
PYTHON_BIN="${PYTHON_BIN:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python}"

export PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export DIFFLET_BACKEND="${DIFFLET_BACKEND:-trainium}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-4}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"
export DIFFLET_RUN_HUNYUAN_VIDEO_NUMERICAL="${DIFFLET_RUN_HUNYUAN_VIDEO_NUMERICAL:-1}"
export DIFFLET_HUNYUAN_VIDEO_SOURCE_DIR="${DIFFLET_HUNYUAN_VIDEO_SOURCE_DIR:-/home/ubuntu/.cache/huggingface/hub/hunyuanvideo-real}"
export DIFFLET_HUNYUAN_VIDEO_COMPILED_DIR="${DIFFLET_HUNYUAN_VIDEO_COMPILED_DIR:-${REPO_DIR}/.difflet-cache/hunyuan_n4_20d40s2r/compiled}"
export DIFFLET_HUNYUAN_VIDEO_BUNDLE="${DIFFLET_HUNYUAN_VIDEO_BUNDLE:-${REPO_DIR}/.difflet-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors}"
export DIFFLET_HUNYUAN_VIDEO_NUMERICAL_METRICS="${DIFFLET_HUNYUAN_VIDEO_NUMERICAL_METRICS:-/tmp/difflet_hunyuan_video_trajectory_metrics.json}"
export DIFFLET_HUNYUAN_VIDEO_MIN_COSINE="${DIFFLET_HUNYUAN_VIDEO_MIN_COSINE:-0.999}"

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m pytest tests/numerical/test_hunyuan_video_vs_diffusers.py -q "$@"
