#!/bin/bash
# usage: cell.sh <name> <cp> <cp_mode> <num_layers> [extra preflight args]
W=/home/ubuntu/qwen-v2-work
name=$1; cp=$2; mode=$3; nl=$4; shift 4
ref=/home/ubuntu/qwen-v1-work/ref-l4.pt
[ "$nl" = 60 ] && ref=/home/ubuntu/qwen-v1-work/ref-l60.pt
[ "$nl" = 30 ] && ref=/home/ubuntu/qwen-v2-work/ref-l30.pt
mkdir -p $W/$name
bash $W/dev.sh /home/ubuntu/.cache/difflet-neuron-qwen --reference $ref --num-layers $nl \
  --exec-mode compile --cp-degree $cp --cp-mode $mode \
  --save $W/$name/out.pt --out-json $W/$name/result.json "$@"
