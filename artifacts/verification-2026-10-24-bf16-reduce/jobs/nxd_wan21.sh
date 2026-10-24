#!/usr/bin/env bash
# Task 13 comparison: Wan2.1 on the NxD backend (this branch: bf16 all-reduce,
# DiT + umT5), the Wan2.1 report's config: 480x832x9, 50 steps, g5.0, seed 42,
# host VAE. Work dirs kept for the latents.
source /home/ubuntu/campaign/bf16-reduce/jobs/env.sh
base=$CAMPAIGN/runs/nxd_wan21
mkdir -p "$base"
cd /home/ubuntu
common=(--model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision "$WAN21_REV" --tp-degree 4 --cp-degree 1
        --height 480 --width 832 --num-frames 9)
if [ ! -e "$base/compile.done" ]; then
  timed nxd_wan21_compile "$base/compile.log" difflet compile "${common[@]}" --host-vae || exit 1
  touch "$base/compile.done"
fi
for name in warmup warm1 warm2 warm3; do
  dir=$base/$name
  mkdir -p "$dir"
  timed "nxd_wan21_$name" "$dir/generate.log" difflet generate "${common[@]}" --host-vae \
    --steps 50 --guidance-scale 5.0 --seed 42 --prompt "$PROMPT" --output "$dir/out.mp4" \
    --work-dir "$dir/work" --keep-work-dir || exit 1
done
touch "$base/measure.done"
