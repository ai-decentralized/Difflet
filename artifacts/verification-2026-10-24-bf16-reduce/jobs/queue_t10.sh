#!/usr/bin/env bash
# Device queue for Task 9/10 after the Qwen compile: LTX-2 compiles, then the
# Qwen and LTX-2 fp32/bf16 measurement runs. Each step is skipped if its
# marker exists, so a restart resumes where it stopped.
source /home/ubuntu/campaign/bf16-reduce/jobs/env.sh
cd /home/ubuntu
if [ ! -e "$CAMPAIGN/runs/qwen/compile.done" ]; then
  bash "$CAMPAIGN/jobs/qwen_compile.sh"
  [ -e "$CAMPAIGN/runs/qwen/compile.done" ] || { echo "queue: qwen compile FAILED"; exit 1; }
fi
echo "queue: qwen compile done"

mkdir -p "$CAMPAIGN/runs/ltx2"
if [ ! -e "$CAMPAIGN/runs/ltx2/compile.done" ]; then
  for dt in fp32 bf16; do
    timed "ltx2_compile_$dt" "$CAMPAIGN/runs/ltx2/compile_$dt.log" \
      difflet compile --model-id Lightricks/LTX-2 --revision "$LTX_REV" \
        --tp-degree 4 --cp-degree 1 --height 480 --width 704 --num-frames 49 \
        --tp-reduce-dtype "$dt" || { echo "queue: ltx2 compile $dt FAILED"; exit 1; }
  done
  touch "$CAMPAIGN/runs/ltx2/compile.done"
fi
echo "queue: ltx2 compile done"

for model in qwen ltx2; do
  for dt in fp32 bf16; do
    if [ ! -e "$CAMPAIGN/runs/$model/$dt/measure.done" ]; then
      bash "$CAMPAIGN/jobs/measure.sh" "$model" "$dt" full || { echo "queue: measure $model $dt FAILED"; exit 1; }
    fi
    echo "queue: measured $model $dt"
  done
done
touch "$CAMPAIGN/runs/queue_t10.done"
