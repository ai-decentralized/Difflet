#!/usr/bin/env bash
# Experiment E1 on Wan (the const-max lane of the attn-constmax-models run): steps <arm>:<mode>, in
# order, on one device and its CPU slice, each under the device's lock and only once fuser shows
# the device idle. Settings: scripts/quant_bench/bench.sh's wan21_official / wan22_official keys
# (480x832x81, CFG, the model cards' negative prompts, snapshots pinned).
#   arms: A bf16 (shipped attention_cte), B A + DIFFLET_EXP_ATTN_CONSTMAX=1, C --quant mxfp8,
#         D C + DIFFLET_EXP_ATTN_CONSTMAX=1
#   modes: compile (untimed load + warm-up), smoke (the gate's 10-step generate, latents kept),
#          timed (one process, two identical requests)
# One cache for every arm and both models (NEFFs are keyed by their HLO): ROOT/wan-cache. Run it
# from a snapshot of a commit (git archive, TREE.txt naming the commit), so edits other lanes make
# to a shared checkout cannot reach an arm.
#
# usage: arms_wan.sh <wan21|wan22> <device> <arm:mode>...
# env: ROOT (default /mnt/nvme/difflet-exp/constmax-models), PY, LOCKS (the device lock directory)
set -u
MODEL=$1 DEV=$2; shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROOT=${ROOT:-/mnt/nvme/difflet-exp/constmax-models}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
LOCKS=${LOCKS:-/home/ubuntu/.claude/jobs/3bceca85/tmp/devlocks}
CORES=$((4 * DEV))-$((4 * DEV + 3))
CPUS=$((48 + 6 * DEV))-$((53 + 6 * DEV)),$((144 + 6 * DEV))-$((149 + 6 * DEV))
C=$ROOT/wan-cache
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export HF_HUB_OFFLINE=1 NEURON_RT_VISIBLE_CORES=$CORES NEURON_NUMA_PIN=0 PYTHONPATH=$REPO
unset DIFFLET_QUANT_TARGETS DIFFLET_COMPILE_CACHE TORCH_NEURONX_NEFF_CACHE_DIR \
  TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
  DIFFLET_NEURON_WEIGHT_STORE DIFFLET_SHARED_WEIGHTS_DIR OMP_NUM_THREADS MKL_NUM_THREADS \
  DIFFLET_EXP_ATTN_CONSTMAX
eval "$(grep -E '^WAN2[12]_NEG=' "$REPO/scripts/quant_bench/bench.sh")"  # the cards' bytes
PROMPT="a cat walking through a garden"
case $MODEL in
  wan21) M=(--model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68
            --height 480 --width 832 --num-frames 81 --tp-degree 4)
    STEPS=50 G=(--guidance-scale 5.0) NEG=$WAN21_NEG ;;
  wan22) M=(--model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7
            --dual-expert --height 480 --width 832 --num-frames 81 --tp-degree 4)
    STEPS=40 G=(--guidance-scale 4.0 --guidance-scale-2 3.0) NEG=$WAN22_NEG ;;
  *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac

run() {  # run <arm> <mode>
  local arm=$1 mode=$2 Q=() E=() W=$ROOT/$MODEL/$1
  case $arm in
    A) ;; B) E=(DIFFLET_EXP_ATTN_CONSTMAX=1) ;; C) Q=(--quant mxfp8) ;;
    D) Q=(--quant mxfp8); E=(DIFFLET_EXP_ATTN_CONSTMAX=1) ;;
    *) echo "unknown arm $arm" >&2; return 2 ;;
  esac
  mkdir -p "$W"
  while [ -n "$(fuser "/dev/neuron$DEV" 2>/dev/null)" ]; do sleep 30; done  # another session's
  { date '+%F %T'; echo "tree $(cat "$REPO/TREE.txt" 2>/dev/null || git -C "$REPO" rev-parse --short HEAD) dirty(difflet)=$(git -C "$REPO" status --porcelain -- difflet 2>/dev/null | wc -l)"
    echo "model $MODEL arm $arm mode $mode cores $CORES cpus $CPUS env ${E[*]:-none} quant ${Q[*]:-none}"
    echo "fuser /dev/neuron$DEV: '$(fuser "/dev/neuron$DEV" 2>&1)'"
    mpstat -P "$CPUS" 1 1 | tail -n +4; uptime; df -h /mnt/nvme | tail -1; } > "$W/$mode.env.txt" 2>&1
  local cmd=(env "${E[@]}" /usr/bin/time -v taskset -c "$CPUS" "$PY" -m difflet.cli.main)
  if [ "$mode" = compile ]; then
    cmd+=(compile --backend neuron "${M[@]}" "${Q[@]}" --cache-dir "$C")
  elif [ "$mode" = smoke ]; then
    cmd+=(generate --backend neuron "${M[@]}" "${Q[@]}" --cache-dir "$C" --steps 10 --seed 42 "${G[@]}"
          --negative-prompt "$NEG" --prompt "$PROMPT" --output "$W/smoke.mp4" --work-dir "$W/work_smoke"
          --keep-work-dir)
  else
    : > "$W/requests.jsonl"
    for i in 1 2; do
      "$PY" -c 'import json, sys
print(json.dumps({"prompt": sys.argv[1], "seed": 42, "steps": int(sys.argv[2]), "output": sys.argv[3],
                  "negative_prompt": sys.argv[4]}, ensure_ascii=False))' "$PROMPT" "$STEPS" "$W/warm_$i.mp4" "$NEG" \
        >> "$W/requests.jsonl"
    done
    cmd+=(generate --backend neuron "${M[@]}" "${Q[@]}" --cache-dir "$C" --steps "$STEPS" --seed 42 "${G[@]}"
          --negative-prompt "$NEG" --requests "$W/requests.jsonl" --work-dir "$W/work" --keep-work-dir)
  fi
  printf '%q ' "${cmd[@]}" > "$W/$mode.cmd.txt"; echo >> "$W/$mode.cmd.txt"
  ( cd "$REPO" && flock "$LOCKS/neuron$DEV.lock" "${cmd[@]}" 2>&1 | "$PY" "$REPO/scripts/quant_bench/stamp.py"
    exit "${PIPESTATUS[0]}" ) > "$W/$mode.log" 2>&1
  local rc=$?
  echo "exit $rc $(date '+%F %T')" >> "$W/$mode.env.txt"
  echo "$(date '+%F %T') $MODEL $arm $mode exit $rc" >> "$ROOT/$MODEL/progress.txt"
  return $rc
}

mkdir -p "$ROOT/$MODEL"
for step in "$@"; do run "${step%%:*}" "${step#*:}" || echo "$(date '+%F %T') $MODEL $step FAILED" >> "$ROOT/$MODEL/progress.txt"; done
