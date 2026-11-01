#!/usr/bin/env bash
# Phase 5: JSONL batch (--requests, 2 requests) at TP2xCFG2, then each request as a single run.
# usage: p5.sh <step ...>   steps: batch single0 single1
set -u
CAMP=/home/ubuntu/.claude/jobs/294a04fc/tmp/campaign
RC="python3 $CAMP/runcell.py"
M=hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
SHAPE="--height 320 --width 512 --num-frames 61"
D=$CAMP/cells/p5-batch
L2C="--tp-degree 2 --cfg-parallel"
rm -f $CAMP/p45/p5.done
for s in "$@"; do
  case $s in
    batch)
      [ -f $D/batch.meta.json ] && continue
      echo "[p5] batch $(date +%T)"
      $RC --name p5-batch/batch --backend trainium --env HF_HUB_OFFLINE=1 -- generate --model-id $M $SHAPE --steps 50 \
          $L2C --requests $D/requests.jsonl --keep-work-dir --work-dir $D/batch.work;;
    single0)
      [ -f $D/single0.meta.json ] && continue
      echo "[p5] single0 $(date +%T)"
      $RC --name p5-batch/single0 --backend trainium --env HF_HUB_OFFLINE=1 -- generate --model-id $M $SHAPE --steps 50 \
          $L2C --prompt "a cinematic shot of a red fox running through a snowy forest" \
          --negative-prompt "blurry, low quality, distorted, watermark" --seed 42 \
          --keep-work-dir --work-dir $D/single0.work --output $D/single0.mp4;;
    single1)
      [ -f $D/single1.meta.json ] && continue
      echo "[p5] single1 $(date +%T)"
      $RC --name p5-batch/single1 --backend trainium --env HF_HUB_OFFLINE=1 -- generate --model-id $M $SHAPE --steps 50 \
          $L2C --prompt "a red panda eating bamboo in a misty bamboo forest, soft morning light" --seed 1234 \
          --keep-work-dir --work-dir $D/single1.work --output $D/single1.mp4;;
    *) echo "unknown step $s";;
  esac
done
echo "P5_DONE $* $(date +%T)" | tee $CAMP/p45/p5.done
