#!/usr/bin/env bash
# The MXFP8 attention plan's Task 6 (docs/plans/2026-11-14-attn-quant-model-runs.md): one model
# key's arms at its Plan A profile shape on one lane (one idle device, one CPU set), with
# scripts/quant_bench/bench.sh's commands (args_for: shape, steps, flags, prompts, seed 42;
# arm_flags: bf16, mx, both = --quant mxfp8 --attn-quant qk+pv, attn = --attn-quant qk+pv).
#
# usage: run_arms.sh <step> <key> <arm> <device> <cpus>
#   warm   one generate at 2 steps: compiles and warms the arm (untimed)
#   timed  one generate process, two identical requests (--requests): the step time and spread
#   gen2   one more generate, one request, in a new process: the determinism check
#   lane <key> <arm> <device> <cpus>: timed bf16, timed <arm>, timed mx when steps.py's G3 row
#          asks for the plan's one linear-alone run, then gen2 of <arm>
# Into $ROOT/<key>/<arm>/: <step>.log (bench.sh's timed: stamped lines, /usr/bin/time -v, "exit
# <rc>"), the outputs (warm, timed_1, timed_2, gen2 .<ext>), work_<step>/ kept (latents), md5.txt;
# $ROOT/<key>/runs.log: each step's start (with the checkout, its HEAD and "dirty") and end with
# uptime and every device's fuser. Before each step `fuser /dev/neuron<device>` must find the
# device and no process on it (BR4), else nothing runs (exit 3); a step first removes its own
# earlier outputs.
# env: ROOT (/mnt/nvme/difflet-mxattn/task6), PY. --cache-dir: the NVMe mirror $ROOT/cache/<fp8|
#      video> of Plan A's per-key root cache (ruling R7; NEFF/HLO copied, quantized dirs, bf16
#      copies and weight stores symlinked, read by hits); hubs: hub() below (run_all.sh's
#      hub_of, plus `qwen_image both`, which run_all.sh has no lane for).
set -u
MINE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$MINE/../quant_bench/bench.sh"  # args_for, arm_flags, timed, request_line; its env hygiene
ROOT=${ROOT:-/mnt/nvme/difflet-mxattn/task6}
export PATH=$PATH:/opt/aws/neuron/bin

STEP=${1:-} KEY=${2:-} ARM=${3:-} DEV=${4:-} CPUS=${5:-}
if [ -z "$CPUS" ]; then
  echo "usage: run_arms.sh <warm|timed|gen2|lane> <key> <arm> <device> <cpus>" >&2
  exit 2
fi
case $STEP in warm|timed|gen2|lane) ;; *) echo "run_arms.sh: unknown step $STEP" >&2; exit 2 ;; esac
case $DEV in ''|*[!0-9]*) echo "run_arms.sh: device must be a number, got $DEV" >&2; exit 2 ;; esac
DEV=$((10#$DEV))  # 08 is device 8, not an octal error
QT=; args_for "$KEY"; arm_flags "$ARM"
L=$ROOT/$KEY
W=$L/$ARM
mkdir -p "$W"
log() { echo "$(date '+%F %T') $*" >> "$L/runs.log"; }
fusers() {  # every device's fuser exit: 1 = no process holds it
  local d out=
  for d in $(seq 0 15); do fuser "/dev/neuron$d" > /dev/null 2>&1; out+=" $d:$?"; done
  echo "$out"
}

if [ "$STEP" = lane ]; then
  "$0" timed "$KEY" bf16 "$DEV" "$CPUS" || exit
  "$0" timed "$KEY" "$ARM" "$DEV" "$CPUS" || exit
  if [ "$ARM" = both ]; then
    row=$("$PY" "$MINE/steps.py" --g3 "$KEY" "$L/bf16/timed.log" "$W/timed.log") || exit
    log "G3 $row"
    if grep -q '"linear_run_needed": true' <<< "$row"; then
      "$0" timed "$KEY" mx "$DEV" "$CPUS" || exit
    fi
  fi
  exec "$0" gen2 "$KEY" "$ARM" "$DEV" "$CPUS"
fi

hub() {  # run_all.sh's hub_of: the hub the arm's cache entries are keyed by
  case "$KEY $ARM" in
    "qwen_image mx"|"qwen_image both"|ltx_2\ *|wan21_official\ *|wan22_official\ *)
      echo "$HOME/.cache/huggingface/hub" ;;
    *) echo /mnt/nvme/difflet-quant-bench-hub ;;
  esac
}
export HF_HUB_CACHE=$(hub)
CACHE=$ROOT/cache/video
if [ "$KEY" = qwen_image ]; then CACHE=$ROOT/cache/fp8; fi
# not in the compile-cache key; saves Wan bf16 ~7 min of NKI tracing per process (Plan A)
case "$KEY $ARM" in wan2[12]_official\ bf16) export DIFFLET_NKI_TRACE_CACHE=1 ;; esac
export NEURON_RT_VISIBLE_CORES=$((4 * DEV))-$((4 * DEV + 3))

GEN=(env "${QENV[@]}" "$PY" -m difflet.cli.main generate --backend neuron "${M[@]}" "${QF[@]}"
     --cache-dir "$CACHE")
case $STEP in
  warm) P[1]=2  # args_for's P starts --steps <STEPS>
        GEN+=("${P[@]}" --prompt "$PROMPT" --output "$W/warm.$EXT") ;;
  timed) : > "$W/requests.jsonl"
         for i in 1 2; do request_line "$W/timed_$i.$EXT" >> "$W/requests.jsonl"; done
         GEN+=("${P[@]}" --requests "$W/requests.jsonl") ;;
  gen2) GEN+=("${P[@]}" --prompt "$PROMPT" --output "$W/gen2.$EXT") ;;
esac
GEN+=(--work-dir "$W/work_$STEP" --keep-work-dir)

held=$(fuser "/dev/neuron$DEV" 2>&1)
rc=$?
# idle: exit 1 and no output; fuser also exits 1 for a device that does not exist, with a message
if [ $rc -ne 1 ] || [ -n "$held" ]; then
  log "$KEY $ARM $STEP: /dev/neuron$DEV fuser exit $rc${held:+ ($held)}: BUSY or missing, not run"
  exit 3
fi
dirty=$(git -C "$REPO" status --porcelain 2>/dev/null)
log "$KEY $ARM $STEP start: device $DEV cores $NEURON_RT_VISIBLE_CORES cpus $CPUS" \
  "tree $REPO $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo no-git)${dirty:+ dirty}" \
  "($(taskset -c "$CPUS" nproc) threads) hub $HF_HUB_CACHE cache $CACHE;" \
  "df: $(df --output=avail -BG / /mnt/nvme | tail -2 | tr -s ' \n' ' ');" \
  "uptime:$(uptime | sed 's/.*load/ load/'); fuser:$(fusers)"
rm -rf "$W/work_$STEP" "$W/$STEP"*."$EXT"  # no earlier output of this step passes for this run's
timed "$W/$STEP.log" "${GEN[@]}"
rc=$?
log "$KEY $ARM $STEP end: exit $rc; uptime:$(uptime | sed 's/.*load/ load/'); fuser:$(fusers)"
( cd "$W" && md5sum ./*."$EXT"; find . -name 'latents*.pt' -exec md5sum {} + ) > "$W/md5.txt" 2>&1
exit $rc
