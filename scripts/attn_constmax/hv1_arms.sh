#!/usr/bin/env bash
# Experiment E1, lane HV1: one arm of HunyuanVideo 1.0 at 544x960x129, tp4 (Plan A's
# hunyuan_video_L key: bench.sh's hunyuan_video flags at that shape, 50 steps, seed 42, guidance
# 6.0): its compile (load + warm-up, untimed), its 10-step gate generate, or its timed generate
# (one process, two identical requests; the second request is the warm one).
#
# usage: hv1_arms.sh <A|B|C|D> <compile|smoke|timed> <device>
#   A: bf16, shipped attention_cte          B: A + DIFFLET_EXP_ATTN_CONSTMAX=1
#   C: --quant mxfp8 (the default targets)  D: C + DIFFLET_EXP_ATTN_CONSTMAX=1
# All four arms share one cache root (ROOT/cache, seeded from Plan A's NVMe mirror of
# ~/difflet-cache-video), so B and D compile only their new block graphs. The device command runs
# under the device's lock (BORROW=1: flock -n, for a device that is not the lane's) and refuses a
# device that another process holds; cores 4d..4d+3, the device's 12-thread CPU slice,
# NEURON_NUMA_PIN=0. <mode>.env.txt logs the tree, fuser and mpstat.
# env: ROOT (default /mnt/nvme/difflet-exp/constmax-models/hunyuan_video), PY, BORROW
set -u
# Shared host: the HY4 kernel campaign takes corenode.lock for its whole-node runs; every arm's
# device run takes it too, so the two workloads wait for each other instead of colliding.
if [ -z "${CORENODE_LOCKED:-}" ]; then
  export CORENODE_LOCKED=1
  exec flock /home/ubuntu/ws/trainium-kernel-agent/locks/corenode.lock "$0" "$@"
fi
ARM=$1 MODE=$2 DEV=$3
LOCK=/home/ubuntu/.claude/jobs/3bceca85/tmp/devlocks/neuron$DEV.lock
if [ -z "${HV1_LOCKED:-}" ]; then
  [ -n "${BORROW:-}" ] && NB=-n || NB=
  HV1_LOCKED=1 exec flock $NB -E 75 "$LOCK" "$0" "$@"
fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROOT=${ROOT:-/mnt/nvme/difflet-exp/constmax-models/hunyuan_video}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
CORES=$((4 * DEV))-$((4 * DEV + 3))
CPUS=$((48 + 6 * DEV))-$((53 + 6 * DEV)),$((144 + 6 * DEV))-$((149 + 6 * DEV))
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export HF_HUB_CACHE=/mnt/nvme/difflet-quant-bench-hub HF_HUB_OFFLINE=1
export NEURON_RT_VISIBLE_CORES=$CORES NEURON_NUMA_PIN=0 PYTHONPATH=$REPO
unset DIFFLET_QUANT_TARGETS DIFFLET_COMPILE_CACHE TORCH_NEURONX_NEFF_CACHE_DIR \
  TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
  DIFFLET_NEURON_WEIGHT_STORE DIFFLET_SHARED_WEIGHTS_DIR OMP_NUM_THREADS MKL_NUM_THREADS \
  DIFFLET_EXP_ATTN_CONSTMAX
M=(--model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773
   --height 544 --width 960 --num-frames 129 --tp-degree 4)
G=(--seed 42 --guidance-scale 6.0)
PROMPT="a cinematic shot of a red fox running through a snowy forest"
Q=()
case $ARM in
  A) ;;
  B) export DIFFLET_EXP_ATTN_CONSTMAX=1 ;;
  C) Q=(--quant mxfp8) ;;
  D) export DIFFLET_EXP_ATTN_CONSTMAX=1; Q=(--quant mxfp8) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
C=$ROOT/cache W=$ROOT/$ARM
mkdir -p "$C" "$W"
held=$(fuser "/dev/neuron$DEV" 2>/dev/null)
{ date '+%F %T'; echo "arm $ARM mode $MODE device $DEV cores $CORES cpus $CPUS"
  echo "head $(git -C "$REPO" rev-parse --short HEAD)"; git -C "$REPO" status --porcelain -- difflet
  md5sum "$REPO"/difflet/backends/neuron/ops_impl/attention*.py \
    "$REPO"/difflet/models/hunyuan_video/modeling_hunyuan_video.py
  env | grep -E '^(DIFFLET|NEURON|HF_HUB|PYTHONPATH)' | sort
  echo "fuser /dev/neuron$DEV: [$held]"; mpstat -P "$CPUS" 1 1 | tail -n +4; } > "$W/$MODE.env.txt" 2>&1
if [ "$DEV" != 0 ] && [ -n "$held" ]; then
  echo "refused: /dev/neuron$DEV held by $held" | tee -a "$W/$MODE.env.txt" >&2; exit 3
fi
cd "$REPO"
run() {  # run <timeout> <log> <args...>
  local t=$1 log=$2; shift 2
  /usr/bin/time -v timeout "$t" taskset -c "$CPUS" "$PY" -m difflet.cli.main "$@" > "$log" 2>&1
}
case $MODE in
  compile) run 4h "$W/compile.log" compile --backend neuron "${M[@]}" "${Q[@]}" --cache-dir "$C" ;;
  smoke) run 2h "$W/smoke.log" generate --backend neuron "${M[@]}" "${Q[@]}" --cache-dir "$C" \
           --steps 10 "${G[@]}" --prompt "$PROMPT" --output "$W/smoke.mp4" \
           --work-dir "$W/work_smoke" --keep-work-dir ;;
  timed) : > "$W/requests.jsonl"
    for i in 1 2; do
      "$PY" -c 'import json, sys; print(json.dumps({"prompt": sys.argv[1], "seed": 42, "steps": 50, "output": sys.argv[2]}))' \
        "$PROMPT" "$W/warm_$i.mp4" >> "$W/requests.jsonl"
    done
    run 3h "$W/timed.log" generate --backend neuron "${M[@]}" "${Q[@]}" --cache-dir "$C" \
      --steps 50 "${G[@]}" --requests "$W/requests.jsonl" --work-dir "$W/work" --keep-work-dir ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
rc=$?
echo "exit $rc $(date '+%F %T')" >> "$W/$MODE.env.txt"
exit $rc
