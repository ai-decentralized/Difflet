#!/usr/bin/env bash
# Compile Qwen-Image fp32-reduce (legacy key) then bf16-reduce, TP4.
source /home/ubuntu/campaign/bf16-reduce/jobs/env.sh
cd /home/ubuntu
mkdir -p "$CAMPAIGN/runs/qwen"
for dt in fp32 bf16; do
  timed "qwen_compile_$dt" "$CAMPAIGN/runs/qwen/compile_$dt.log" \
    difflet compile --model-id Qwen/Qwen-Image --revision "$QWEN_REV" \
      --tp-degree 4 --cp-degree 1 --height 1024 --width 1024 --tp-reduce-dtype "$dt" || exit 1
done
touch "$CAMPAIGN/runs/qwen/compile.done"
