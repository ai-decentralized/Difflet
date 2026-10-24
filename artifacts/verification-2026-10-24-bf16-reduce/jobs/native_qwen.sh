#!/usr/bin/env bash
# Task 13: Qwen-Image on the PyTorch-native backend (origin/feat/torch-native-backend
# + the env-gated latent-dump/step-timer hooks, uncommitted), TP4, 1024^2, 20
# steps, g4.0, seed 42. Run 'compile' starts on an empty NEFF cache (first-use
# compile); then warmup (discarded), warm1..3, and a page-cold run.
source /home/ubuntu/campaign/bf16-reduce/jobs/env.sh
NS=/home/ubuntu/campaign/native_src
NV=/home/ubuntu/venvs/native-312
NC=$CAMPAIGN/native_cache/qwen
base=$CAMPAIGN/runs/native_qwen
mkdir -p "$base"
cd /home/ubuntu

run() {  # run <name> [cold]
  local name=$1 dir=$base/$1 start end rc
  mkdir -p "$dir"
  if [ "${2:-}" = cold ]; then sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; fi
  start=$(date +%s.%N)
  HF_HUB_OFFLINE=1 DIFFLET_NO_JEMALLOC=1 DIFFLET_NEURON_PG_TIMEOUT_S=3600 PYTHONSAFEPATH=1 \
  DIFFLET_COMPILE_CACHE=$NC PYTHONPATH=$NS PATH=$NV/bin:$PATH \
  DIFFLET_DUMP_LATENTS_DIR=$dir DIFFLET_STEP_TIMES_FILE=$dir/steps.jsonl \
    "$NV/bin/python" -m difflet.cli.main generate --backend neuron --exec-mode compile \
      --model-id Qwen/Qwen-Image --revision "$QWEN_REV" --tp-degree 4 --cp-degree 1 \
      --height 1024 --width 1024 --steps 20 --guidance-scale 4.0 --seed 42 \
      --prompt "$PROMPT" --output "$dir/out.png" --cache-dir "$NC" \
      2>&1 | python -u "$CAMPAIGN/jobs/tstamp.py" > "$dir/generate.log"
  rc=${PIPESTATUS[0]}
  end=$(date +%s.%N)
  echo "native_qwen_$name seconds=$(echo "$end - $start" | bc) rc=$rc" | tee -a "$CAMPAIGN/timings.txt"
  return $rc
}

[ -e "$base/compile/out.png" ] || { rm -rf "$NC"; run compile || exit 1; }
run warmup || exit 1
for i in 1 2 3; do run "warm$i" || exit 1; done
run cold cold || exit 1
touch "$base/measure.done"
