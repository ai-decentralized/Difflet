#!/usr/bin/env bash
# Task 10/11 measurement for one model x reduce dtype.
#   measure.sh <model: qwen|ltx2|flux|wan22|hv> <dtype: fp32|bf16> <mode: full|quick>
# full : 1 warm-up (discarded, page cache warmed) + 3 timed warm runs + 1 page-cold run
# quick: 1 warm-up + 1 timed warm run (regression check)
# Every run dumps latents (DIFFLET_DUMP_LATENTS_DIR) and per-step times
# (DIFFLET_STEP_TIMES_FILE) into its own run dir, output written as out.png/out.mp4.
source /home/ubuntu/campaign/bf16-reduce/jobs/env.sh
model=$1 dt=$2 mode=${3:-full}
case $model in
  qwen) id=Qwen/Qwen-Image rev=$QWEN_REV ext=png
        args=(--height 1024 --width 1024 --steps 20 --guidance-scale 4.0) ;;
  ltx2) id=Lightricks/LTX-2 rev=$LTX_REV ext=mp4
        args=(--height 480 --width 704 --num-frames 49 --steps 20 --guidance-scale 1.0) ;;
  flux) id=black-forest-labs/FLUX.1-dev rev=$FLUX_REV ext=png
        args=(--height 1024 --width 1024 --steps 28 --guidance-scale 3.5) ;;
  wan22) id=Wan-AI/Wan2.2-T2V-A14B-Diffusers rev=$WAN22_REV ext=mp4
        args=(--height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0) ;;
  hv)   id=hunyuanvideo-community/HunyuanVideo rev=$HV_REV ext=mp4
        args=(--height 320 --width 512 --num-frames 61 --steps 20 --guidance-scale 6.0) ;;
  *) echo "unknown model $model"; exit 2 ;;
esac
base=$CAMPAIGN/runs/$model/$dt
mkdir -p "$base"
cd /home/ubuntu

run() {  # run <name> [cold]
  local name=$1 dir=$base/$1
  mkdir -p "$dir"
  if [ "${2:-}" = cold ]; then sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; fi
  local start end rc
  start=$(date +%s.%N)
  DIFFLET_DUMP_LATENTS_DIR=$dir DIFFLET_STEP_TIMES_FILE=$dir/steps.jsonl \
    difflet generate --model-id "$id" --revision "$rev" --tp-degree 4 --cp-degree 1 \
      --tp-reduce-dtype "$dt" "${args[@]}" --seed 42 --prompt "$PROMPT" \
      --output "$dir/out.$ext" 2>&1 | python -u "$CAMPAIGN/jobs/tstamp.py" > "$dir/generate.log"
  rc=${PIPESTATUS[0]}
  end=$(date +%s.%N)
  echo "${model}_${dt}_$name seconds=$(echo "$end - $start" | bc) rc=$rc" | tee -a "$CAMPAIGN/timings.txt"
  return $rc
}

if [ "$mode" = extra ]; then  # extra <run name>: one more warm run after a page-cache warm-up
  run warmup_extra || exit 1
  run "${4:-warm4}" || exit 1
  exit 0
fi
run warmup || exit 1
if [ "$mode" = full ]; then
  for i in 1 2 3; do run "warm$i" || exit 1; done
  run cold cold || exit 1
else
  run warm1 || exit 1
fi
touch "$base/measure.done"
