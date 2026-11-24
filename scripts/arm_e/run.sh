#!/usr/bin/env bash
# Arm E (docs/plans/2026-11-22-mxfp8-constmax-teacache-trn3.md): one step of one model's lane on
# one device. Arm E is the 2026-11-18 arm A command (model flags as run then, from
# 10af3dcc^:scripts/attn_constmax/{arms,arms_wan,hv1_arms,hv15_arms,qwen_lane,ltx2_lane}.sh; the
# requests are arm A's file) plus --quant mxfp8 --attention const-max and TeaCache.
# HunyuanVideo-1.5 has no MXFP8 (D4): --attention const-max only, decoding on one core like arm A.
#
# usage: run.sh <model> <E1|E2|F|Fb> <compile|smoke_d|smoke_tc|timed> <device 0-15>
#   models    qwen_image flux ltx_2 wan21 wan22 hunyuan_video hunyuan_video_15
#   E1        --teacache-online-delta 0.6          E2  --teacache-cadence 2
#   F         E2 at the model's best layout: + --sp (qwen_image, flux, hunyuan_video: tp4 + SP);
#             wan21 / wan22: --tp-degree 2 --cfg-parallel --sp (tp2 x CFG2 + SP, 4 cores);
#             ltx_2: --tp-degree 2 --cfg-parallel (tp2 x CFG2)
#   Fb        F without --quant mxfp8, in the bf16 + const-max cache (README: MXFP8 with SP
#             measured 1.92x slower on HunyuanVideo); F arms keep their own smoke dirs
#   compile   difflet compile with E's flags (load + warm-up; TeaCache is a generate flag)
#   smoke_d   10 steps, E's flags without TeaCache (11-18's arm D on this tree)
#   smoke_tc  14 steps (wan22: 20) with --teacache-cadence 2
#   timed     one process, arm A's two requests (hunyuan_video: with the cat prompt, D5)
# Outputs: OUT/<model>/<arm>/ (compile, timed) and OUT/<model>/<smoke_d|smoke_tc>/: <mode>.log
# (timestamped by stamp.py), <mode>.env.txt (snapshot, slice, uptime and NEFF count before and
# after, exit code). The caches are 11-18's, reused in place.
# Exit: the command's (124: timeout); 2 bad arguments; 3 refused (a previous log, a missing
# input, the disk); 75 the device or its CPU slice is busy (wait and call again).
# env: SRC (required: the code snapshot directory, the run's PYTHONPATH and cwd), OUT (default
#   /mnt/nvme/difflet-exp/arm-e), PY, HF_HUB_CACHE (default /mnt/nvme/difflet-quant-bench-hub,
#   the same snapshots as the default hub four of the 11-18 lanes read), TIMEOUT (default per
#   model and mode, about twice 11-18's arm D), DRY_RUN=1 (print the command, its environment
#   and the requests; run nothing, write nothing)
set -u
die() { echo "run.sh: $*" >&2; exit 2; }
[ $# -eq 4 ] || die "usage: run.sh <model> <E1|E2|F|Fb> <compile|smoke_d|smoke_tc|timed> <device 0-15>"
MODEL=$1 ARM=$2 MODE=$3 DEV=$4
BASE=/mnt/nvme/difflet-exp/constmax-models  # 11-18: arm A's requests and the caches
OUT=${OUT:-/mnt/nvme/difflet-exp/arm-e}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
Q=(--quant mxfp8 --attention const-max) G=() EXT=mp4 STEPS=50 TC_STEPS=14 PROMPT=
HUB=/mnt/nvme/difflet-quant-bench-hub X=() FX=(--sp) CB= S=
# A: arm A's directory; C: the cache; T: timeouts of compile, a smoke, timed
case $MODEL in
  qwen_image) M=(--model-id Qwen/Qwen-Image --height 1024 --width 1024 --tp-degree 4)
    G=(--guidance-scale 7.5) A=A C=qwen_image/cache/D CB=qwen_image/cache/B EXT=png T=(30m 10m 10m) ;;
  flux) M=(--model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024 --tp-degree 4)
    A=bf16 C=flux/cache/flux-mxcm CB=flux/cache/flux-cm EXT=png T=(30m 10m 10m) ;;
  ltx_2) M=(--model-id Lightricks/LTX-2 --height 512 --width 768 --num-frames 121 --tp-degree 4)
    A=bf16 C=ltx_2/cache/mxcm CB=../constmax/cache/ltx_2-cm T=(2h 30m 30m)
    FX=(--tp-degree 2 --cfg-parallel) ;;  # LTX-2 has no SP; tp2 loads and warms up longer
  wan21) M=(--model-id Wan-AI/Wan2.1-T2V-14B-Diffusers
            --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68
            --height 480 --width 832 --num-frames 81 --tp-degree 4)
    G=(--guidance-scale 5.0) A=A C=wan-cache CB=wan-cache T=(2h 20m 40m)
    FX=(--tp-degree 2 --cfg-parallel --sp)  # the later --tp-degree wins
    # 11-18's hub: the bf16 copy's manifest, and so the MXFP8 digest, holds the source's path
    HUB=$HOME/.cache/huggingface/hub ;;
  wan22) M=(--model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers
            --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7
            --dual-expert --height 480 --width 832 --num-frames 81 --tp-degree 4)
    G=(--guidance-scale 4.0 --guidance-scale-2 3.0) A=A C=wan-cache STEPS=40 TC_STEPS=20
    T=(2h 20m 40m) HUB=$HOME/.cache/huggingface/hub CB=wan-cache
    FX=(--tp-degree 2 --cfg-parallel --sp) ;;
  hunyuan_video) M=(--model-id hunyuanvideo-community/HunyuanVideo
                    --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773
                    --height 544 --width 960 --num-frames 129 --tp-degree 4)
    G=(--guidance-scale 6.0) A=A C=hunyuan_video/cache CB=hunyuan_video/cache PROMPT="a cat walking through a garden"
    T=(4h 30m 1h) ;;
  hunyuan_video_15) M=(--model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
                       --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf
                       --height 480 --width 848 --num-frames 121 --vae-cores 1)
    Q=(--attention const-max) A=A C=hunyuan_video_15/cache T=(2h 20m 40m) ;;
  *) die "unknown model $MODEL" ;;
esac
case $ARM in
  E1) TC=(--teacache-online-delta 0.6) ;;
  E2) TC=(--teacache-cadence 2) ;;
  F|Fb) [ -n "$CB" ] || die "arm $ARM: $MODEL has no SP layout here"
    TC=(--teacache-cadence 2) X=("${FX[@]}") S=_$ARM
    [ "$ARM" = F ] || { Q=(--attention const-max); C=$CB; } ;;
  *) die "unknown arm $ARM" ;;
esac
case $MODE in
  compile) W=$OUT/$MODEL/$ARM TO=${T[0]} ;;
  smoke_d) W=$OUT/$MODEL/$MODE$S TO=${T[1]} N=10 TC=() ;;
  smoke_tc) W=$OUT/$MODEL/$MODE$S TO=${T[1]} N=$TC_STEPS TC=(--teacache-cadence 2) ;;
  timed) W=$OUT/$MODEL/$ARM TO=${T[2]} N=$STEPS ;;
  *) die "unknown mode $MODE" ;;
esac
case $DEV in [0-9]|1[0-5]) ;; *) die "unknown device $DEV" ;; esac
[ -n "${SRC:-}" ] || die "SRC (the code snapshot directory) is not set"
TO=${TIMEOUT:-$TO} C=$BASE/$C AREQ=$BASE/$MODEL/$A/requests.jsonl
CORES=$((4 * DEV))-$((4 * DEV + 3))
if [ "$DEV" -lt 8 ]; then
  CPUS=$((48 + 6 * DEV))-$((53 + 6 * DEV)),$((144 + 6 * DEV))-$((149 + 6 * DEV))
else  # NUMA node 0
  CPUS=$((6 * DEV - 48))-$((6 * DEV - 43)),$((6 * DEV + 48))-$((6 * DEV + 53))
fi
[ -f "$AREQ" ] && [ -d "$C" ] || { echo "missing $AREQ or $C" >&2; exit 3; }

field() {  # a field of arm A's first request ("" when it has none)
  "$PY" -c 'import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").readline()).get(sys.argv[2], ""), end="")' \
    "$AREQ" "$1"
}
requests() {  # arm A's requests with "output" rewritten (hunyuan_video: and the prompt)
  "$PY" -c 'import json, os, sys
src, w, prompt = sys.argv[1:]
lines = open(src, encoding="utf-8").read().splitlines()
assert len(lines) == 2, f"{src}: {len(lines)} requests, not 2"
for i, line in enumerate(lines, 1):
    r = json.loads(line)
    r["output"] = "%s/warm_%d%s" % (w, i, os.path.splitext(r["output"])[1])
    r["prompt"] = prompt
    print(json.dumps(r, ensure_ascii=False))' "$AREQ" "$W" "$PROMPT"
}
PROMPT=${PROMPT:-$(field prompt)}
NEG=$(field negative_prompt)
[ -z "$NEG" ] || G+=(--negative-prompt "$NEG")

cmd=(/usr/bin/time -v timeout "$TO" taskset -c "$CPUS" "$PY" -m difflet.cli.main)
if [ "$MODE" = compile ]; then
  cmd+=(compile --backend neuron "${M[@]}" "${Q[@]}" "${X[@]}" --cache-dir "$C")
else
  cmd+=(generate --backend neuron "${M[@]}" "${Q[@]}" "${X[@]}" --cache-dir "$C" --steps "$N" --seed 42
        "${G[@]}")
  if [ "$MODE" = timed ]; then cmd+=(--requests "$W/requests.jsonl")
  else cmd+=(--prompt "$PROMPT" --output "$W/smoke.$EXT"); fi
  cmd+=(--work-dir "$W/work" --keep-work-dir "${TC[@]}")
fi

export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=${HF_HUB_CACHE:-$HUB}
export NEURON_RT_VISIBLE_CORES=$CORES NEURON_NUMA_PIN=0 PYTHONPATH=$SRC
# the 11-18 scripts' list, and the CPU-only jobs' switch: exported by a caller, it would keep
# torch from loading the neuron backend
unset DIFFLET_QUANT_TARGETS DIFFLET_COMPILE_CACHE TORCH_NEURONX_NEFF_CACHE_DIR \
  TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
  DIFFLET_NEURON_WEIGHT_STORE DIFFLET_SHARED_WEIGHTS_DIR OMP_NUM_THREADS MKL_NUM_THREADS \
  DIFFLET_EXP_ATTN_CONSTMAX DIFFLET_STEP_TIMES_FILE TORCH_DEVICE_BACKEND_AUTOLOAD
# after the unset: nproc obeys OMP_NUM_THREADS
[ "$(taskset -c "$CPUS" nproc)" = 12 ] || { echo "cpu slice $CPUS is not 12 threads" >&2; exit 3; }
# 11-18's Qwen-Image lane timed with the per-step file on
[ "$MODEL $MODE" != "qwen_image timed" ] || export DIFFLET_STEP_TIMES_FILE=$W/step_times.jsonl

CORENODE=/home/ubuntu/ws/trainium-kernel-agent/locks/corenode.lock
LOCKS=/home/ubuntu/.claude/jobs/3bceca85/tmp/devlocks
[ -d "$LOCKS" ] || LOCKS=/mnt/nvme/difflet-exp/arm-e/devlocks
sha=$(cat "$SRC/.arm_e_sha" 2>/dev/null || echo unknown)
neffs() { find "$C" -name '*.neff' | wc -l; }
describe() {
  echo "model $MODEL arm $ARM mode $MODE device $DEV cores $CORES cpus $CPUS timeout $TO"
  echo "snapshot $SRC sha $sha"
  echo "locks $LOCKS/neuron$DEV.lock (exclusive), $CORENODE (shared)"
  env | grep -E '^(DIFFLET|NEURON|HF_HUB|PYTHONPATH|TORCH|OMP_|MKL_)' | sort
  echo "cache $C neffs $(neffs)"
}
if [ -n "${DRY_RUN:-}" ]; then
  describe
  echo "cwd $SRC log $W/$MODE.log"
  [ "$MODE" != timed ] || { echo "requests $W/requests.jsonl:"; requests || exit 3; }
  printf '%q ' "${cmd[@]}"; echo "| taskset -c $CPUS $PY $SRC/scripts/quant_bench/stamp.py"
  exit 0
fi

[ -d "$SRC/difflet" ] && [ -f "$SRC/scripts/quant_bench/stamp.py" ] \
  || { echo "no difflet/ or scripts/quant_bench/stamp.py in $SRC" >&2; exit 3; }
[ ! -e "$W/$MODE.log" ] || { echo "$W/$MODE.log exists: move the run away first" >&2; exit 3; }
[ "$(df -BG --output=avail /mnt/nvme | tail -1 | tr -dc 0-9)" -ge 100 ] \
  || { echo "less than 100 GB free on /mnt/nvme" >&2; exit 3; }
mkdir -p "$W" "$LOCKS"
exec 8> "$LOCKS/neuron$DEV.lock"
flock -n 8 || { echo "busy: $LOCKS/neuron$DEV.lock is held" >&2; exit 75; }

holders() {  # pid, its NEURON_RT_VISIBLE_CORES, its command, for every holder of the device
  for p in $(fuser "/dev/neuron$DEV" 2>/dev/null); do
    v=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^NEURON_RT_VISIBLE_CORES=//p')
    echo "$p cores=${v:-unset} $(ps -o args= -p "$p" | cut -c1-100)"
  done
}
overlap() {  # the holders pinned to one of this device's cores (every runtime opens neuron0 too)
  holders | while read -r p c rest; do
    for r in $(tr , ' ' <<< "${c#cores=}"); do
      [ "$r" != unset ] && [ "${r%%-*}" -le $((4 * DEV + 3)) ] && [ "${r##*-}" -ge $((4 * DEV)) ] \
        && { echo "$p $c $rest"; break; }
    done
  done
}
E=$W/$MODE.env.txt
{ date '+%F %T'; describe; uptime; df -h /mnt/nvme | tail -1
  echo "holders of /dev/neuron$DEV:"; holders; LC_ALL=C mpstat -P "$CPUS" 2 1; } > "$E" 2>&1
busy=$(overlap)
# every CPU of the slice at least 90% idle over 2 s
grep '^Average: *[0-9]' "$E" | awk '$NF < 90 { bad = 1 } END { exit bad || NR != 12 }' \
  || busy+="${busy:+; }cpus $CPUS below 90% idle"
if [ -n "$busy" ]; then
  echo "busy: $busy" | tee -a "$E" >&2
  exit 75
fi

if [ "$MODE" = timed ]; then
  requests > "$W/requests.jsonl" || exit 3
  rm -f "$W/step_times.jsonl"
fi
printf '%q ' "${cmd[@]}" > "$W/$MODE.cmd.txt"; echo >> "$W/$MODE.cmd.txt"
( cd "$SRC" && flock -s "$CORENODE" "${cmd[@]}" 2>&1 \
    | taskset -c "$CPUS" "$PY" "$SRC/scripts/quant_bench/stamp.py"
  exit "${PIPESTATUS[0]}" ) > "$W/$MODE.log" 2>&1
rc=$?
{ echo "neffs after $(neffs)"; uptime; echo "exit $rc $(date '+%F %T')"; } >> "$E"
exit $rc
