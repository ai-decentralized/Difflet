#!/bin/bash
# End-to-end A/B of the neuron attention exp engine: one checkout, each model generated twice back
# to back, DIFFLET_NEURON_ATTENTION_EXP=vector and =scalar, same seed and flags. Two models at a
# time, one trn3 device (TP4 = 4 LNC=2 logical cores) per run: at most four devices busy.
#
#   benchmark/trn3/attention_exp/ab_e2e.sh OUT_DIR [FIRST_DEVICE]
#
# FIRST_DEVICE (default 10) and the three after it are used. Needs the models in the HF cache
# (HF_HUB_CACHE, offline). Afterwards: python benchmark/trn3/attention_exp/compare.py OUT_DIR
set -u
OUT=$(realpath -m "${1:?usage: ab_e2e.sh OUT_DIR [FIRST_DEVICE]}")
DEV0=${2:-10}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$OUT"
P="a red fox walking through fresh snow in a pine forest, golden hour light"

declare -A ARGS=(
  [flux]="black-forest-labs/FLUX.1-dev --height 1024 --width 1024 --steps 50"
  [qwen]="Qwen/Qwen-Image --height 1024 --width 1024 --steps 50"
  [wan]="Wan-AI/Wan2.2-T2V-A14B-Diffusers --height 480 --width 832 --num-frames 81 --steps 40"
  [hv15]="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --height 480 --width 848 --num-frames 121 --steps 30"
  [hv]="hunyuanvideo-community/HunyuanVideo --height 544 --width 960 --num-frames 61 --steps 30"
  [ltx]="Lightricks/LTX-2 --height 512 --width 768 --num-frames 121 --steps 30"
)
declare -A EXT=([flux]=png [qwen]=png [wan]=mp4 [hv15]=mp4 [hv]=mp4 [ltx]=mp4)

run() {  # model exp device
  local m=$1 exp=$2 dev=$3 name="$1_$2"
  local cores="$((4 * dev))-$((4 * dev + 3))"
  read -r -a a <<< "${ARGS[$m]}"
  (
    cd "$ROOT"
    export PYTHONPATH=$ROOT DIFFLET_BACKEND=neuron PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
    export NEURON_RT_VISIBLE_CORES=$cores DIFFLET_NEURON_ATTENTION_EXP=$exp
    t0=$(date +%s)
    python -m difflet.cli.main generate --model-id "${a[0]}" --backend neuron --tp-degree 4 \
      --seed 42 --prompt "$P" "${a[@]:1}" --output "$OUT/$name.${EXT[$m]}" \
      --cache-dir "$OUT/cache" --work-dir "$OUT/work_$name" > "$OUT/$name.log" 2>&1
    echo "EXIT=$? wall=$(( $(date +%s) - t0 ))s" >> "$OUT/$name.log"
  ) &
}

{
  echo "commit $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || cat "$ROOT/REVISION" 2>/dev/null)"
  echo "host $(hostname) $(date -u +%FT%TZ)"
  python -c "import torch, torch_neuronx; print('torch', torch.__version__, 'torch-neuronx', torch_neuronx.__version__)"
  python -c "import neuronxcc; print('neuronx-cc', neuronxcc.__version__)" 2>/dev/null
  pip list 2>/dev/null | grep -iE "^(nki|nkilib|diffusers|transformers) "
} > "$OUT/env.txt"

for pair in "flux qwen" "wan hv15" "hv ltx"; do
  read -r m1 m2 <<< "$pair"
  run "$m1" vector "$DEV0"; run "$m1" scalar "$((DEV0 + 1))"
  run "$m2" vector "$((DEV0 + 2))"; run "$m2" scalar "$((DEV0 + 3))"
  wait
done
echo DONE >> "$OUT/env.txt"
