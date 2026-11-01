#!/usr/bin/env bash
# R26: native TP2xCFG2 (native-312 venv), 320x512x61, 50 steps, seed 42, --teacache-cadence 2.
# The no-TeaCache reference is Phase 0's native-tp2cfg/gen50 (same command without the flag).
set -u
CAMP=/home/ubuntu/.claude/jobs/294a04fc/tmp/campaign
D=$CAMP/cells/native-tp2cfg
rm -f $CAMP/p45/native_cad2.done
if [ ! -f $D/cad2.meta.json ]; then
  python3 $CAMP/runcell.py --name native-tp2cfg/cad2 --backend neuron --env HF_HUB_OFFLINE=1 \
      --env PATH=/home/ubuntu/venvs/native-312/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin -- \
      generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --height 320 --width 512 \
      --num-frames 61 --steps 50 --seed 42 --tp-degree 2 --cfg-parallel \
      --prompt "a cinematic shot of a red fox running through a snowy forest" --teacache-cadence 2 \
      --keep-work-dir --work-dir $D/cad2.work --output $D/cad2.mp4
fi
echo "NATIVE_CAD2_DONE $(date +%T)" | tee $CAMP/p45/native_cad2.done
