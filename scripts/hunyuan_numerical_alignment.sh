#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/nova}"
PYTHON_BIN="${PYTHON_BIN:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python}"

export PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export NOVA_BACKEND="${NOVA_BACKEND:-trainium}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-4}"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_RT_VIRTUAL_CORE_SIZE:-2}"
export NOVA_RUN_HUNYUAN_VIDEO_NUMERICAL="${NOVA_RUN_HUNYUAN_VIDEO_NUMERICAL:-1}"
export NOVA_HUNYUAN_VIDEO_SOURCE_DIR="${NOVA_HUNYUAN_VIDEO_SOURCE_DIR:-/home/ubuntu/.cache/huggingface/hub/hunyuanvideo-real}"
export NOVA_HUNYUAN_VIDEO_COMPILED_DIR="${NOVA_HUNYUAN_VIDEO_COMPILED_DIR:-${REPO_DIR}/.nova-cache/hunyuan_n4_20d40s2r/compiled}"
export NOVA_HUNYUAN_VIDEO_BUNDLE="${NOVA_HUNYUAN_VIDEO_BUNDLE:-${REPO_DIR}/.nova-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors}"
export NOVA_HUNYUAN_VIDEO_NUMERICAL_METRICS="${NOVA_HUNYUAN_VIDEO_NUMERICAL_METRICS:-/tmp/nova_hunyuan_video_trajectory_metrics.json}"
export NOVA_HUNYUAN_VIDEO_MIN_COSINE="${NOVA_HUNYUAN_VIDEO_MIN_COSINE:-0.999}"

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m pytest tests/numerical/test_hunyuan_video_vs_diffusers.py -q "$@"
