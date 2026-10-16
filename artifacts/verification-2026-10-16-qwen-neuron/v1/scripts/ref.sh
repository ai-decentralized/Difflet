#!/bin/bash
# usage: ref.sh <num_layers> <out> [extra args]
WT=/home/ubuntu/Difflet/.claude/worktrees/qwen-image-neuron
SNAP=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6
exec env HF_HUB_OFFLINE=1 PYTHONPATH=$WT DIFFLET_BACKEND=neuron \
  /home/ubuntu/workspace/native_venv/bin/python $WT/tests/manual/qwen_image_dit_reference.py \
  --model-dir $SNAP --num-layers "$1" --timesteps 1000,500 \
  --embeds $WT/artifacts/verification-2026-10-16-qwen-neuron/prompt_embeds.pt --out "$2" "${@:3}"
