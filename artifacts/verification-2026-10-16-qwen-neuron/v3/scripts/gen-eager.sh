#!/bin/bash
# usage: gen.sh <run_dir> <probe> <steps> <tp> <cp> <cp_mode> [seed]
run="$1"; probe="$2"; steps="$3"; tp="$4"; cp="$5"; mode="$6"; seed="${7:-42}"
exec bash /home/ubuntu/qwen-v3-work/cli.sh "$run" "$probe" generate --backend neuron --exec-mode eager \
  --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
  --tp-degree "$tp" --cp-degree "$cp" --cp-mode "$mode" --height 1024 --width 1024 \
  --steps "$steps" --seed "$seed" \
  --prompt "A red fox sitting in a snowy forest at sunrise, highly detailed photograph" \
  --output "$run/out.png" --cache-dir /home/ubuntu/.cache/difflet-neuron-qwen \
  --work-dir "$run/work" --keep-work-dir
