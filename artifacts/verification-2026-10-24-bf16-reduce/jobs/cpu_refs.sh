#!/usr/bin/env bash
# Task 12 step 1: CPU fp32 references for Qwen-Image and LTX-2, started from
# the device runs' initial noise (identical across fp32/bf16, checked).
# A 1-step smoke per model first (times one CPU forward), then the full run.
source /home/ubuntu/campaign/bf16-reduce/jobs/env.sh
PY=/home/ubuntu/venvs/difflet-neuron/bin/python
SCRIPT=$REPO/scripts/cpu_fp32_reference.py
snap() { ls -d "/home/ubuntu/.cache/huggingface/hub/models--$1/snapshots/$2"; }
QWEN_PATH=$(snap Qwen--Qwen-Image "$QWEN_REV")
LTX_PATH=$(snap Lightricks--LTX-2 "$LTX_REV")
out=$CAMPAIGN/runs/cpu_ref
mkdir -p "$out"
for spec in "qwen_image:$QWEN_PATH:$CAMPAIGN/runs/qwen/fp32/warm1" "ltx_2:$LTX_PATH:$CAMPAIGN/runs/ltx2/fp32/warm2"; do
  IFS=: read -r model path init <<< "$spec"
  if [ ! -e "$out/$model/reference.json" ]; then
    timed "cpu_ref_${model}_smoke" "$out/${model}_smoke.log" \
      "$PY" "$SCRIPT" --model "$model" --model-path "$path" --initial-latents-dir "$init" \
        --out-dir "$out/${model}_smoke" --steps 2 || { echo "cpu_ref smoke $model FAILED"; exit 1; }
    timed "cpu_ref_${model}" "$out/${model}.log" \
      "$PY" "$SCRIPT" --model "$model" --model-path "$path" --initial-latents-dir "$init" \
        --out-dir "$out/$model" || { echo "cpu_ref $model FAILED"; exit 1; }
  fi
done
touch "$out/done"
