#!/usr/bin/env bash
# E2 model level: Wan 2.1 480x832x81 with scripts/quant_bench/bench.sh's wan21_official flags
# (tp4, seed 42, CFG 5.0, the model card's negative prompt, 50 steps), one arm per
# DIFFLET_EXP_ATTN_SPARSE value.
# usage: wan_arms.sh <compile|gate|timed> <arm> <device> <cpus>
#   arm: bf16 (unset: shipped attention_cte) | full (temporal:20: every tile, the routing gate)
#        | t6 (temporal:6, density 0.534) | t10 (temporal:10, density 0.756)
#   compile  difflet compile into $ROOT/cache/<arm>; a non-bf16 arm links bf16's weights-bf16
#            copy when it is complete (the copies are identical; disk)
#   gate     one 10-step generate: $ROOT/<arm>/gate.mp4, its latents, md5.txt
#   timed    one generate process, two identical 50-step requests: timed_{1,2}.mp4, timed.log
# Before each step `fuser /dev/neuron<device>` must find no process (else exit 3).
set -u
STEP=$1 ARM=$2 DEV=$3 CPUS=$4
ROOT=${ROOT:-/mnt/nvme/difflet-exp/sparse/model}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
case $ARM in
  bf16) unset DIFFLET_EXP_ATTN_SPARSE ;;
  full) export DIFFLET_EXP_ATTN_SPARSE=temporal:20 ;;
  t6) export DIFFLET_EXP_ATTN_SPARSE=temporal:6 ;;
  t10) export DIFFLET_EXP_ATTN_SPARSE=temporal:10 ;;
  *) echo "wan_arms.sh: unknown arm $ARM" >&2; exit 2 ;;
esac
# bench.sh's flags and request line, read from it (sourcing it would run its arms)
eval "$(sed -n '/^WAN21_NEG=/p;/^args_for() {/,/^}/p;/^request_line() {/,/^}/p' "$REPO/scripts/quant_bench/bench.sh")"
args_for wan21_official
# bench.sh's environment hygiene; the hub run_arms.sh used for wan21_official
unset DIFFLET_COMPILE_CACHE TORCH_NEURONX_NEFF_CACHE_DIR TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR \
  TORCH_NEURONX_HLO_CACHE_DIR TORCHINDUCTOR_CACHE_DIR DIFFLET_NEURON_WEIGHT_STORE \
  DIFFLET_SHARED_WEIGHTS_DIR OMP_NUM_THREADS MKL_NUM_THREADS DIFFLET_QUANT_TARGETS
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=$HOME/.cache/huggingface/hub DIFFLET_NKI_TRACE_CACHE=1
export NEURON_RT_VISIBLE_CORES=$((4 * DEV))-$((4 * DEV + 3)) NEURON_NUMA_PIN=0
C=$ROOT/cache/$ARM W=$ROOT/$ARM
mkdir -p "$C" "$W"
log() { echo "$(date '+%F %T') $ARM $STEP $*" >> "$ROOT/runs.log"; }
if fuser "/dev/neuron$DEV" > /dev/null 2>&1; then log "device $DEV busy: not run"; exit 3; fi
log "start dev=$DEV cpus=$CPUS spec=${DIFFLET_EXP_ATTN_SPARSE:-unset} head=$(git -C "$REPO" rev-parse --short HEAD) idle%=$(mpstat -P "$CPUS" 1 1 | awk '/^Average/ && $2 ~ /^[0-9]+$/ {s += $NF; n++} END {printf "%.0f", s / n}')"
GEN=("$PY" -m difflet.cli.main generate --backend neuron "${M[@]}" --cache-dir "$C")
cd "$REPO"
case $STEP in
  compile)
    if [ "$ARM" != bf16 ] && [ ! -e "$C/weights-bf16" ]; then
      ln -s "$ROOT/cache/bf16/weights-bf16" "$C/weights-bf16"
    fi
    /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main compile --backend neuron "${M[@]}" \
      --cache-dir "$C" > "$W/compile.log" 2>&1 ;;
  gate)
    P[1]=10  # args_for's P starts --steps <STEPS>
    /usr/bin/time -v taskset -c "$CPUS" "${GEN[@]}" "${P[@]}" --prompt "$PROMPT" \
      --output "$W/gate.mp4" --work-dir "$W/work_gate" --keep-work-dir > "$W/gate.log" 2>&1 ;;
  timed)
    : > "$W/requests.jsonl"
    for i in 1 2; do request_line "$W/timed_$i.mp4" >> "$W/requests.jsonl"; done
    /usr/bin/time -v taskset -c "$CPUS" "${GEN[@]}" "${P[@]}" --requests "$W/requests.jsonl" \
      --work-dir "$W/work_timed" --keep-work-dir > "$W/timed.log" 2>&1 ;;
  *) echo "wan_arms.sh: unknown step $STEP" >&2; exit 2 ;;
esac
rc=$?
(cd "$W" && find . -maxdepth 3 \( -name '*.mp4' -o -name 'latents*.pt' \) | sort | xargs -r md5sum > md5.txt)
log "end exit=$rc $(grep -h 'step median' "$W/$STEP.log" 2>/dev/null | tr '\n' ' ')"
exit $rc
