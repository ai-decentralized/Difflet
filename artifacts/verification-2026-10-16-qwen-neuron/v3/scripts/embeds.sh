#!/bin/bash
WT=/home/ubuntu/Difflet/.claude/worktrees/qwen-image-neuron
SNAP=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6
start=$(date +%s.%N)
env PATH=/home/ubuntu/workspace/native_venv/bin:$PATH HF_HUB_OFFLINE=1 PYTHONPATH=$WT \
  /home/ubuntu/workspace/native_venv/bin/python $WT/tests/manual/check_qwen_image_prompt_embeds.py \
  --model-dir $SNAP --prompt "A red fox sitting in a snowy forest at sunrise, highly detailed photograph" \
  --fp32 --device-embeds /home/ubuntu/qwen-v3-work/tp4-s50-gen1/embeds_rank0.pt \
  --out-json /home/ubuntu/qwen-v3-work/embeds_parity.json
echo "EXIT rc=$? wall_s=$(echo "$(date +%s.%N) - $start" | bc)"
