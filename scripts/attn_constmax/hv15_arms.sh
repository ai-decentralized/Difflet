#!/usr/bin/env bash
# Const-max at model level, HunyuanVideo-1.5 lane: one arm's compile (untimed; loads and warms
# up), its 10-step gate generate, or its timed generate (one process, two identical requests),
# with scripts/quant_bench/bench.sh's hv15_official flags (480x848x121, 50 steps, seed 42, the
# snapshot's guider: true CFG 6.0, default layout tp2 x CFG2 = one device). All four arms share
# one compile cache (NEFFs are keyed by HLO, the bf16 copy and the quantized directory by content;
# the native path keeps no args-keyed artifact), so B and D only compile their new block graphs.
#
# usage: hv15_arms.sh <A|B|C|D> <compile|smoke|timed> <cores> <cpus>
#   A bf16 (shipped cte); B A + DIFFLET_EXP_ATTN_CONSTMAX=1; C --quant mxfp8 with bench.sh's
#   HV15_MX_TARGETS (the former default G1 + G3: no MXFP8 default exists); D C + const-max
# env: ROOT (default /mnt/nvme/difflet-exp/constmax-models/hunyuan_video_15), PY
set -u
ARM=$1 MODE=$2 CORES=$3 CPUS=$4
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROOT=${ROOT:-/mnt/nvme/difflet-exp/constmax-models/hunyuan_video_15}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export HF_HUB_OFFLINE=1 NEURON_RT_VISIBLE_CORES=$CORES NEURON_NUMA_PIN=0 PYTHONPATH=$REPO
unset DIFFLET_QUANT_TARGETS DIFFLET_COMPILE_CACHE TORCH_NEURONX_NEFF_CACHE_DIR \
  TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
  DIFFLET_NEURON_WEIGHT_STORE DIFFLET_SHARED_WEIGHTS_DIR OMP_NUM_THREADS MKL_NUM_THREADS \
  DIFFLET_EXP_ATTN_CONSTMAX
M=(--model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
   --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf --height 480 --width 848 --num-frames 121)
PROMPT="a cat walking through a garden"
MX='transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v);transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)'
Q=()
case $ARM in
  A) ;;
  B) export DIFFLET_EXP_ATTN_CONSTMAX=1 ;;
  C) export DIFFLET_QUANT_TARGETS=$MX; Q=(--quant mxfp8) ;;
  D) export DIFFLET_QUANT_TARGETS=$MX DIFFLET_EXP_ATTN_CONSTMAX=1; Q=(--quant mxfp8) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
C=$ROOT/cache W=$ROOT/$ARM
mkdir -p "$C" "$W"
d=$(( ${CORES%%-*} / 4 ))
{ date '+%F %T'; echo "head $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -- difflet | wc -l)"
  git -C "$REPO" status --porcelain -- difflet
  echo "arm $ARM cores $CORES cpus $CPUS threads $(taskset -c "$CPUS" nproc)"
  env | grep -E '^(DIFFLET_|NEURON_RT_VISIBLE|NEURON_NUMA)' | sort
  fuser "/dev/neuron$d" 2>&1; echo "fuser /dev/neuron$d exit $?"
  mpstat -P "$CPUS" 1 1 | tail -n +4; } > "$W/$MODE.env.txt" 2>&1
cd "$REPO"
run() { /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main "$@"; }
if [ "$MODE" = compile ]; then
  run compile --backend neuron "${M[@]}" "${Q[@]}" --cache-dir "$C" > "$W/compile.log" 2>&1
elif [ "$MODE" = smoke ]; then  # the gate's 10-step generate, latents kept
  run generate --backend neuron "${M[@]}" "${Q[@]}" --cache-dir "$C" --steps 10 --seed 42 \
    --prompt "$PROMPT" --output "$W/smoke.mp4" --work-dir "$W/work_smoke" --keep-work-dir \
    > "$W/smoke.log" 2>&1
else
  : > "$W/requests.jsonl"
  for i in 1 2; do
    "$PY" -c 'import json, sys; print(json.dumps({"prompt": sys.argv[1], "seed": 42, "steps": 50, "output": sys.argv[2]}))' \
      "$PROMPT" "$W/warm_$i.mp4" >> "$W/requests.jsonl"
  done
  run generate --backend neuron "${M[@]}" "${Q[@]}" --cache-dir "$C" --steps 50 --seed 42 \
    --requests "$W/requests.jsonl" --work-dir "$W/work" --keep-work-dir > "$W/timed.log" 2>&1
fi
rc=$?
echo "exit $rc $(date '+%F %T')" >> "$W/$MODE.env.txt"
exit $rc
