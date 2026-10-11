#!/bin/bash
# One warm Wan 2.1 81f tp4 generate with per-line timestamps, per-step timing,
# NRT INFO logs, and a 5 s sampler of host memory + shard page-cache residency.
set -u
cd ~/work/Difflet && source .venv/bin/activate
OUT=${OUT:-$HOME/work/scratch/e2e/run1}
mkdir -p "$OUT"
W=~/.cache/difflet/_shared_weights
( while true; do
    echo "$(date -u +%H:%M:%S) $(free -m | awk '/Mem:/{print "used="$3"M cache="$6"M avail="$7"M"}') tf_cached=$(python3 ~/work/scratch/tools/fincore.py $W/Wan*transformer*tp4__c06*/* | tail -1 | awk '{print $1}')G"
    sleep 5; done ) > "$OUT/mem.log" 2>&1 &
SAMPLER=$!
export DIFFLET_STEP_TIMING=1 DIFFLET_STEP_TIMING_OUT="$OUT/steps.json"
export NEURON_RT_LOG_LEVEL=${NEURON_RT_LOG_LEVEL:-INFO}
difflet generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --cp-degree 1 --cache-dir $HOME/.cache/difflet \
  --prompt 'a cinematic shot of a red fox running through a snowy forest' --steps 50 --seed 42 \
  --output "$OUT/out.mp4" --height 480 --width 832 --num-frames 81 --guidance-scale 5.0 \
  --negative-prompt '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走' \
  --wan-vae-chunked 2>&1 | python3 -u ~/work/scratch/tools/tsline.py > "$OUT/generate.log"
echo "exit=${PIPESTATUS[0]}" >> "$OUT/generate.log"
kill $SAMPLER
