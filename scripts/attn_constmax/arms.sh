#!/usr/bin/env bash
# Experiment E1 model-level arms (docs/plans/2026-11-18-attn-constmax-sparse-experiments.md): one
# arm's compile (untimed; loads and warms up) or its timed generate (one process, two identical
# requests; report.py's warm step is the second request's DiT step median), with
# scripts/quant_bench/bench.sh's flags per model. Unlike bench.sh, each arm keeps its own cache
# (ROOT/cache/<model>-<arm>): nothing is removed.
#
# usage: arms.sh <flux|ltx_2> <bf16|cm|mx|mxcm> <compile|smoke|timed> <cores> <cpus>
#   bf16: shipped attention_cte; cm: DIFFLET_EXP_ATTN_CONSTMAX=1; mx: --quant mxfp8 alone;
#   mxcm: cm + --quant mxfp8
# env: ROOT (default /mnt/nvme/difflet-exp/constmax), CACHE (default ROOT/cache), PY
set -u
MODEL=$1 ARM=$2 MODE=$3 CORES=$4 CPUS=$5
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROOT=${ROOT:-/mnt/nvme/difflet-exp/constmax}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export HF_HUB_OFFLINE=1 NEURON_RT_VISIBLE_CORES=$CORES NEURON_NUMA_PIN=0 PYTHONPATH=$REPO
unset DIFFLET_QUANT_TARGETS DIFFLET_COMPILE_CACHE TORCH_NEURONX_NEFF_CACHE_DIR \
  TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
  DIFFLET_NEURON_WEIGHT_STORE DIFFLET_SHARED_WEIGHTS_DIR OMP_NUM_THREADS MKL_NUM_THREADS \
  DIFFLET_EXP_ATTN_CONSTMAX
case $MODEL in
  flux) M=(--model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024 --tp-degree 4)
    P=(--steps 50 --seed 42); PROMPT="a cat sitting on a bench"; EXT=png ;;
  ltx_2) M=(--model-id Lightricks/LTX-2 --height 512 --width 768 --num-frames 121 --tp-degree 4)
    P=(--steps 50 --seed 42); PROMPT="a cat walking through a garden"; EXT=mp4 ;;
  *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
Q=()
case $ARM in
  bf16) ;;
  cm) export DIFFLET_EXP_ATTN_CONSTMAX=1 ;;
  mx) Q=(--quant mxfp8) ;;
  mxcm) export DIFFLET_EXP_ATTN_CONSTMAX=1; Q=(--quant mxfp8) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
C=${CACHE:-$ROOT/cache}/$MODEL-$ARM W=$ROOT/$MODEL/$ARM
mkdir -p "$C" "$W"
d=$(( ${CORES%%-*} / 4 ))
{ date '+%F %T'; echo "head $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -- difflet | wc -l)"
  git -C "$REPO" status --porcelain -- difflet
  echo "cores $CORES cpus $CPUS"; fuser "/dev/neuron$d" 2>&1; echo "fuser /dev/neuron$d exit $?"
  mpstat -P "$CPUS" 1 1 | tail -n +4; } > "$W/$MODE.env.txt" 2>&1
cd "$REPO"
if [ "$MODE" = compile ]; then
  /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main compile --backend neuron "${M[@]}" "${Q[@]}" \
    --cache-dir "$C" > "$W/compile.log" 2>&1
elif [ "$MODE" = smoke ]; then  # the gate's 10-step generate, latents kept
  /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main generate --backend neuron "${M[@]}" "${Q[@]}" \
    --cache-dir "$C" --steps 10 --seed 42 --prompt "$PROMPT" --output "$W/smoke.$EXT" \
    --work-dir "$W/work_smoke" --keep-work-dir > "$W/smoke.log" 2>&1
else
  : > "$W/requests.jsonl"
  for i in 1 2; do
    "$PY" -c 'import json, sys; print(json.dumps({"prompt": sys.argv[1], "seed": 42, "steps": 50, "output": sys.argv[2]}))' \
      "$PROMPT" "$W/warm_$i.$EXT" >> "$W/requests.jsonl"
  done
  /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main generate --backend neuron "${M[@]}" "${Q[@]}" \
    --cache-dir "$C" "${P[@]}" --requests "$W/requests.jsonl" --work-dir "$W/work" --keep-work-dir \
    > "$W/timed.log" 2>&1
fi
rc=$?
echo "exit $rc $(date '+%F %T')" >> "$W/$MODE.env.txt"
exit $rc
