#!/usr/bin/env bash
# LTX-2 lane of the const-max model sweep (evidence at 3f674fd2):
# arms.sh's LTX-2 flags and modes, with the four arms A bf16, B bf16 + const-max, C --quant mxfp8,
# D mxfp8 + const-max. Outputs under ROOT/ltx_2/<arm> (score.sh's layout); A and B reuse E1's
# caches, C and D get their own.
#
# usage: ltx2_lane.sh <bf16|cm|mx|mxcm> <compile|smoke|timed> <cores> <cpus>
# env: ROOT (default /mnt/nvme/difflet-exp/constmax-models), E1_CACHE, PY
set -u
ARM=$1 MODE=$2 CORES=$3 CPUS=$4
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROOT=${ROOT:-/mnt/nvme/difflet-exp/constmax-models}
E1_CACHE=${E1_CACHE:-/mnt/nvme/difflet-exp/constmax/cache}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export HF_HUB_OFFLINE=1 NEURON_RT_VISIBLE_CORES=$CORES NEURON_NUMA_PIN=0 PYTHONPATH=$REPO
unset DIFFLET_QUANT_TARGETS DIFFLET_COMPILE_CACHE TORCH_NEURONX_NEFF_CACHE_DIR \
  TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
  DIFFLET_NEURON_WEIGHT_STORE DIFFLET_SHARED_WEIGHTS_DIR OMP_NUM_THREADS MKL_NUM_THREADS \
  DIFFLET_EXP_ATTN_CONSTMAX
M=(--model-id Lightricks/LTX-2 --height 512 --width 768 --num-frames 121 --tp-degree 4)
P=(--steps 50 --seed 42); PROMPT="a cat walking through a garden"
Q=()
case $ARM in
  bf16) C=$E1_CACHE/ltx_2-bf16 ;;
  cm) export DIFFLET_EXP_ATTN_CONSTMAX=1; C=$E1_CACHE/ltx_2-cm ;;
  mx) Q=(--quant mxfp8); C=$ROOT/ltx_2/cache/mx ;;
  mxcm) export DIFFLET_EXP_ATTN_CONSTMAX=1; Q=(--quant mxfp8); C=$ROOT/ltx_2/cache/mxcm ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
W=$ROOT/ltx_2/$ARM
mkdir -p "$C" "$W"
d=$(( ${CORES%%-*} / 4 ))
{ date '+%F %T'; echo "head $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -- difflet | wc -l)"
  echo "arm $ARM cache $C cores $CORES cpus $CPUS DIFFLET_EXP_ATTN_CONSTMAX=${DIFFLET_EXP_ATTN_CONSTMAX-unset} quant=${Q[*]-}"
  fuser "/dev/neuron$d" 2>&1; echo "fuser /dev/neuron$d exit $?"
  mpstat -P "$CPUS" 1 1 | tail -n +4; } > "$W/$MODE.env.txt" 2>&1
cd "$REPO"
if [ "$MODE" = compile ]; then
  /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main compile --backend neuron "${M[@]}" "${Q[@]}" \
    --cache-dir "$C" > "$W/compile.log" 2>&1
elif [ "$MODE" = smoke ]; then  # the gate's 10-step generate, latents kept
  /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main generate --backend neuron "${M[@]}" "${Q[@]}" \
    --cache-dir "$C" --steps 10 --seed 42 --prompt "$PROMPT" --output "$W/smoke.mp4" \
    --work-dir "$W/work_smoke" --keep-work-dir > "$W/smoke.log" 2>&1
else
  : > "$W/requests.jsonl"
  for i in 1 2; do
    "$PY" -c 'import json, sys; print(json.dumps({"prompt": sys.argv[1], "seed": 42, "steps": 50, "output": sys.argv[2]}))' \
      "$PROMPT" "$W/warm_$i.mp4" >> "$W/requests.jsonl"
  done
  /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main generate --backend neuron "${M[@]}" "${Q[@]}" \
    --cache-dir "$C" "${P[@]}" --requests "$W/requests.jsonl" --work-dir "$W/work" --keep-work-dir \
    > "$W/timed.log" 2>&1
fi
rc=$?
echo "exit $rc $(date '+%F %T')" >> "$W/$MODE.env.txt"
exit $rc
