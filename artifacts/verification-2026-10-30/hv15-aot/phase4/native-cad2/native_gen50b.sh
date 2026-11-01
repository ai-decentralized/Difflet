#!/usr/bin/env bash
# R26 control: native TP2xCFG2 no-TeaCache 50-step re-run (same command as Phase 0 gen50) to check run-to-run determinism.
set -u
CAMP=/home/ubuntu/.claude/jobs/294a04fc/tmp/campaign
D=$CAMP/cells/native-tp2cfg
rm -f $CAMP/p45/native_gen50b.done
if [ ! -f $D/gen50b.meta.json ]; then
  python3 $CAMP/runcell.py --name native-tp2cfg/gen50b --backend neuron --env HF_HUB_OFFLINE=1 \
      --env PATH=/home/ubuntu/venvs/native-312/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin -- \
      generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --height 320 --width 512 \
      --num-frames 61 --steps 50 --seed 42 --tp-degree 2 --cfg-parallel \
      --prompt "a cinematic shot of a red fox running through a snowy forest" \
      --keep-work-dir --work-dir $D/gen50b.work --output $D/gen50b.mp4
fi
echo "NATIVE_GEN50B_DONE $(date +%T)" | tee $CAMP/p45/native_gen50b.done
