#!/usr/bin/env bash
# Experiment E1 at model level, Qwen-Image lane (evidence at 3f674fd2):
# one arm's compile (untimed; loads and warms up), its 10-step gate generate, or its timed generate
# (one process, two identical requests; the warm step is the second request's DiT step median),
# with scripts/quant_bench/bench.sh's qwen_image flags (1024x1024, tp4, 50 steps, seed 42,
# guidance 7.5, "a cat sitting on a bench"). Each arm keeps its own cache (ROOT/cache/<arm>).
#
# usage: qwen_lane.sh <A|B|C|D> <compile|smoke|timed> [cores] [cpus]
#   A bf16 (shipped attention_cte), B A + DIFFLET_EXP_ATTN_CONSTMAX=1, C --quant mxfp8, D C + const-max
#   cores/cpus default to device 0's: 0-3 and 48-53,144-149.
# env: ROOT (default /mnt/nvme/difflet-exp/constmax-models/qwen_image), PY, HF_HUB_CACHE
#   (default the NVMe hub copy /mnt/nvme/difflet-quant-bench-hub).
# Refuses (exit 3) when a process pinned to any of the arm's cores holds /dev/neuron<d>.
set -u
main() {
ARM=$1 MODE=$2 CORES=${3:-0-3} CPUS=${4:-48-53,144-149}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROOT=${ROOT:-/mnt/nvme/difflet-exp/constmax-models/qwen_image}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/nvme/difflet-quant-bench-hub}
export NEURON_RT_VISIBLE_CORES=$CORES NEURON_NUMA_PIN=0 PYTHONPATH=$REPO
unset DIFFLET_QUANT_TARGETS DIFFLET_COMPILE_CACHE TORCH_NEURONX_NEFF_CACHE_DIR \
  TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
  DIFFLET_NEURON_WEIGHT_STORE DIFFLET_SHARED_WEIGHTS_DIR OMP_NUM_THREADS MKL_NUM_THREADS \
  DIFFLET_EXP_ATTN_CONSTMAX DIFFLET_STEP_TIMES_FILE
M=(--model-id Qwen/Qwen-Image --height 1024 --width 1024 --tp-degree 4)
P=(--steps 50 --seed 42 --guidance-scale 7.5)
PROMPT="a cat sitting on a bench"
Q=()
case $ARM in
  A) ;;
  B) export DIFFLET_EXP_ATTN_CONSTMAX=1 ;;
  C) Q=(--quant mxfp8) ;;
  D) export DIFFLET_EXP_ATTN_CONSTMAX=1; Q=(--quant mxfp8) ;;
  *) echo "unknown arm $ARM" >&2; return 2 ;;
esac
C=$ROOT/cache/$ARM W=$ROOT/$ARM
mkdir -p "$C" "$W"
d=$(( ${CORES%%-*} / 4 )) lo=${CORES%%-*} hi=${CORES##*-}
holders() {  # pid, its NEURON_RT_VISIBLE_CORES, its command, for every holder of /dev/neuron<d>
  for p in $(fuser "/dev/neuron$d" 2>/dev/null); do
    v=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^NEURON_RT_VISIBLE_CORES=//p')
    echo "$p cores=${v:-unset} $(ps -o args= -p "$p" | cut -c1-100)"
  done
}
overlap() {  # holders pinned to a core in [lo, hi]
  holders | while read -r p c rest; do
    c=${c#cores=}; [ "$c" = unset ] && continue
    a=${c%%-*} b=${c##*-}; [ "$a" -le "$hi" ] && [ "$b" -ge "$lo" ] && echo "$p $c $rest"
  done
}
{ date '+%F %T'; echo "head $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -- difflet | wc -l)"
  git -C "$REPO" status --porcelain -- difflet  # the shared worktree's uncommitted files this run imports
  echo "arm $ARM mode $MODE cores $CORES cpus $CPUS DIFFLET_EXP_ATTN_CONSTMAX=${DIFFLET_EXP_ATTN_CONSTMAX:-unset} quant=${Q[*]:-none}"
  echo "holders of /dev/neuron$d:"; holders
  mpstat -P "$CPUS" 1 1 | tail -n +4; uptime; } > "$W/$MODE.env.txt" 2>&1
busy=$(overlap)
if [ -n "$busy" ]; then
  { echo "REFUSED: cores $CORES busy:"; echo "$busy"; } | tee -a "$W/$MODE.env.txt" >&2
  return 3
fi
cd "$REPO"
if [ "$MODE" = compile ]; then
  /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main compile --backend neuron "${M[@]}" "${Q[@]}" \
    --cache-dir "$C" > "$W/compile.log" 2>&1
elif [ "$MODE" = smoke ]; then  # the gate: 10 steps, latents kept
  /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main generate --backend neuron "${M[@]}" "${Q[@]}" \
    --cache-dir "$C" --steps 10 --seed 42 --guidance-scale 7.5 --prompt "$PROMPT" --output "$W/smoke.png" \
    --work-dir "$W/work_smoke" --keep-work-dir > "$W/smoke.log" 2>&1
else
  : > "$W/requests.jsonl"
  for i in 1 2; do
    "$PY" -c 'import json, sys; print(json.dumps({"prompt": sys.argv[1], "seed": 42, "steps": 50, "output": sys.argv[2]}))' \
      "$PROMPT" "$W/warm_$i.png" >> "$W/requests.jsonl"
  done
  rm -f "$W/step_times.jsonl"
  # lines timestamped (bench.sh's stamp.py): the parent's decode+save after the workers is timed
  ( DIFFLET_STEP_TIMES_FILE=$W/step_times.jsonl /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main \
      generate --backend neuron "${M[@]}" "${Q[@]}" --cache-dir "$C" "${P[@]}" --requests "$W/requests.jsonl" \
      --work-dir "$W/work" --keep-work-dir 2>&1 | taskset -c "$CPUS" "$PY" "$REPO/scripts/quant_bench/stamp.py"
    exit "${PIPESTATUS[0]}" ) > "$W/timed.log" 2>&1
fi
rc=$?
echo "exit $rc $(date '+%F %T')" >> "$W/$MODE.env.txt"
return $rc
}
main "$@"; exit
