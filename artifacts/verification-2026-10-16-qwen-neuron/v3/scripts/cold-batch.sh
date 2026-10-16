#!/bin/bash
W=/home/ubuntu/qwen-v3-work
bash $W/cli-cold.sh $W/cold-compile 1 compile --backend neuron --exec-mode compile --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 --tp-degree 4 --cp-degree 1 --height 1024 --width 1024 --cache-dir /home/ubuntu/.cache/difflet-neuron-qwen-v3cold > $W/cold-compile.log 2>&1
bash $W/gen-cold.sh $W/cold-gen 1 50 4 1 gather_kv > $W/cold-gen.log 2>&1
echo BATCH DONE
