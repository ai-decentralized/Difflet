#!/usr/bin/env bash
# The DiT block profile's device run (docs/plans/2026-11-13-dit-block-profile.md, Task 4): the
# configuration table's 13 (key, arm) lanes, each capture.sh on its own idle device and CPU slice
# into <ROOT>/<key>/<arm>/, then analyze.py on each block it kept. Supervised (restarts resume):
#   .claude/skills/difflet-device-verify/scripts/supervise.sh scripts/profile_dit/run_all.sh \
#     /mnt/nvme/difflet-prof/run/RUN_DONE <name>
#
# Every POLL s, lanes in table order:
#   - finished: STATUS FAILED <reason>, or DONE with every kept block analyzed; running: its lock
#     (<lane>/.lock, held by its job and all it starts) is held, also by an earlier run_all's
#     job, so a lane never runs twice; else pending (no STATUS, BUSY, or DONE not analyzed).
#   - a pending lane that has to capture first passes the disk gate (60 GB free on /, else every
#     such lane ends FAILED disk and none starts) and, once per run_all, its prewarm: its root-disk
#     files read through serially (bench.sh's warm_hub); a missing cache entry ends it FAILED
#     cache miss, as the lane would rebuild it on the root disk.
#   - then a fresh `fuser /dev/neuron<d>` scan of d = 0..15: a device no process holds, none of
#     ours, not 0 or 6 and not one the lane was BUSY on in this run (capture.sh's require_idle
#     checks again before each device command); none: the lane queues behind the first to finish.
#   - the run ends, RUN_DONE written (each lane's STATUS and analyses, the baselines), when
#     nothing of ours runs and no pending lane could start: every lane finished, or the rest
#     have no idle device left to try (a lane BUSY on every one ends BUSY, one never started
#     "not started"); a restart retries them.
# CPU slices: 13 lanes at once with no two sharing a CPU leave 192 / 16 = 12 threads per device:
# its own 6 cores and their SMT siblings on its NUMA node (d < 8: node 1, CPUs 48-95 and 144-191;
# d >= 8: node 0, 0-47 and 96-143). The plan's 48-thread sets fit two lanes per node: four at once.
# Baseline (ruling R17): once at least 4 lanes dump, BASELINE_NEFF (the Qwen-Image smoke block) is
# captured once with the lanes' command (lib.sh's capture_neff) on a free idle device into
# <ROOT>/baseline/inrun/result.json; `run_all.sh baseline <device> <name>` captures it alone into
# <ROOT>/baseline/<name>/.
#
# env: ROOT (/mnt/nvme/difflet-prof/run), PY, CAPTURE (capture.sh here), BASELINE_NEFF (empty: no
#      in-run baseline), POLL (10 s), CAPTURE_LOCK (lib.sh); HOME locates the root-disk caches.
set -u
shopt -s nullglob
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib.sh"
ROOT=${ROOT:-/mnt/nvme/difflet-prof/run}
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
CAPTURE=${CAPTURE:-$HERE/capture.sh}
BASELINE_NEFF=${BASELINE_NEFF-/mnt/nvme/difflet-prof/t1-smoke/qwen_image/bf16/blk1/block.neff}
POLL=${POLL:-10}
export PATH=$PATH:/opt/aws/neuron/bin  # neuron-explorer for the baseline, the SDK's last
LANES=(qwen_image:bf16 qwen_image:mxfp8 flux:bf16 flux:mxfp8 ltx_2:bf16 ltx_2:mxfp8
       wan21_official:bf16 wan21_official:mxfp8 wan22_official:bf16 wan22_official:mxfp8
       hunyuan_video_L:bf16 hunyuan_video_L:mxfp8 hv15_official:bf16)
NOT_OURS=" 0 6 "  # held by another session on 2026-11-13: never used
HUB=$HOME/.cache/huggingface/hub
NVME_HUB=/mnt/nvme/difflet-quant-bench-hub

log() {  # log <text...>: a timestamped line in run_all.log and on stdout (supervise's log)
  local line
  line="$(date '+%F %T') $*"
  echo "$line" >> "$ROOT/run_all.log"
  echo "$line"
}

status() { head -1 "$1/STATUS" 2>/dev/null; }

running() { ! flock -n "$1/.lock" true; }  # running <dir>: its job holds its lock

finished() {  # finished <lane dir>: FAILED, or DONE with every kept block analyzed
  local blk
  case $(status "$1") in
    FAILED*) return 0 ;;
    DONE) for blk in "$1"/blk*/; do
            grep -q "^analyze exit" "$blk/analyze.log" 2>/dev/null || return 1
          done ;;
    *) return 1 ;;
  esac
}

dumping() { running "$1" && ! grep -qx "dump exit 0" "$1/dump.log" 2>/dev/null; }

cpus_of() {  # cpus_of <device>: its 12-thread slice (header)
  local lo=$(( ($1 < 8 ? 48 : 0) + 6 * ($1 % 8) ))
  echo "$lo-$((lo + 5)),$((lo + 96))-$((lo + 101))"
}

hub_of() {  # hub_of <key> <arm>: the hub the lane's cache entries are keyed by (checked
            # 2026-11-13): the NVMe copy, but where an entry names root-hub files. Qwen-Image's
            # quantized digest takes the blob names (the NVMe copy has plain files: another digest,
            # a 29 GB re-quantization on /), Wan 2.2's bf16 copies name their root source;
            # LTX-2 and Wan 2.1 are on root only.
  case "$1 $2" in
    "qwen_image mxfp8"|"ltx_2 "*|"wan21_official "*|"wan22_official "*) echo "$HUB" ;;
    *) echo "$NVME_HUB" ;;
  esac
}

hub_but() {  # hub_but <snapshot> <component>...: the snapshot's entries but those components
  local e
  for e in "$1"/*; do
    case " ${*:2} " in *" ${e##*/} "*) ;; *) echo "$e" ;; esac
  done
  [ -d "$1" ] || echo "$1"  # a missing snapshot is a cache miss
}

lane_files() {  # lane_files <key> <arm>: the root-disk files the lane loads (the 2026-11-13
                # warm-up and smoke logs): root-hub components that no cache entry replaces, the
                # quantized transformer, bf16 copies, weight-store entries
  local V=$HOME/difflet-cache-video S=$HOME/difflet-cache-video/neuron/weights
  local L=$HUB/models--Lightricks--LTX-2/snapshots/64f65d09e0b76c96202b5bc09997673184cf9b5d
  local W1=$HUB/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68
  local W2=$HUB/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7
  local Q1=$V/quantized/Wan-AI--Wan2.1-T2V-14B-Diffusers__38ec498c
  local Q2=$V/quantized/Wan-AI--Wan2.2-T2V-A14B-Diffusers__5be7df96
  case "$1 $2" in
    "qwen_image mxfp8")
      hub_but "$HUB/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6" transformer
      echo "$HOME/difflet-cache-fp8/quantized/Qwen--Qwen-Image__75e0b4be/transformer/mxfp8-65db3df4b2e34023" ;;
    "flux mxfp8")
      echo "$V/quantized/black-forest-labs--FLUX.1-dev__3de623fc/transformer/mxfp8-809df193b011ba1f" ;;
    ltx_2\ *)
      if [ "$2" = bf16 ]; then hub_but "$L" text_encoder
      else hub_but "$L" text_encoder transformer
           echo "$V/quantized/Lightricks--LTX-2__64f65d09/transformer/mxfp8-916a153a8eeb2fa3"; fi
      echo "$S/Lightricks--LTX-2__text_encoder__64f65d09__bfloat16__tp4__9aad1a74d1afb88f" ;;
    wan21_official\ *)
      hub_but "$W1" text_encoder transformer
      echo "$S/Wan-AI--Wan2.1-T2V-14B-Diffusers__text_encoder__38ec498c__bfloat16__tp4__91b0beceaabd499e"
      echo "$S/Wan-AI--Wan2.1-T2V-14B-Diffusers__vae__38ec498c__bfloat16__tp1__fa7540f14397355b"
      if [ "$2" = bf16 ]; then echo "$V/weights-bf16/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer"
      else echo "$Q1/transformer/mxfp8-6d28df7cc78386b5"; fi ;;
    wan22_official\ *)
      hub_but "$W2" transformer transformer_2
      echo "$S/Wan-AI--Wan2.2-T2V-A14B-Diffusers__vae__5be7df96__bfloat16__tp1__c4319fcfc5a85228"
      if [ "$2" = bf16 ]; then
        echo "$V/weights-bf16/5be7df9619b54f4e2667b2755bc6a756675b5cd7/transformer"
        echo "$V/weights-bf16/5be7df9619b54f4e2667b2755bc6a756675b5cd7/transformer_2"
      else echo "$Q2/transformer/mxfp8-372c52a68c440756"
           echo "$Q2/transformer_2/mxfp8-03f6d63f26aab3cf"; fi ;;
    "hunyuan_video_L mxfp8")
      echo "$V/quantized/hunyuanvideo-community--HunyuanVideo__e8c2aaa6/transformer/mxfp8-46e0952c1c5a9f74" ;;
    hv15_official\ *)
      echo "$S/hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v__transformer__286be7ce__bfloat16__tp2__f620128f4bf3c6b4" ;;
  esac
}

prewarm() {  # prewarm <lane dir> <key> <arm>: the lane's root-disk files read to /dev/null;
             # returns 1 (FAILED cache miss written) when one is missing
  local A=$1 p paths missing=() t0=$SECONDS
  mapfile -t paths < <(lane_files "$2" "$3")
  [ ${#paths[@]} -gt 0 ] || return 0
  for p in "${paths[@]}"; do [ -e "$p" ] || missing+=("$p"); done
  if [ ${#missing[@]} -gt 0 ]; then
    write_status "$A" "FAILED cache miss ${missing[*]}"
    log "$2 $3: FAILED cache miss ${missing[*]}"
    return 1
  fi
  log "prewarm $2 $3: ${paths[*]}"
  find -L "${paths[@]}" -type f -exec cat {} + > /dev/null
  log "prewarm $2 $3: $((SECONDS - t0)) s"
}

scan() {  # scan: the devices no process holds and none of ours holds (a running job's), but 0 and 6
  local d x ours=" "
  for x in "$ROOT"/*/*/; do
    if [ -f "$x/device" ] && running "$x"; then ours+="$(< "$x/device") "; fi
  done
  for d in $(seq 0 15); do
    case "$NOT_OURS$ours" in *" $d "*) continue ;; esac
    fuser "/dev/neuron$d" > /dev/null 2>&1 || printf ' %s' "$d"
  done
}

pick() {  # pick <dir>: the first device of SCAN that <dir> was not BUSY on
  local d
  for d in $SCAN; do
    grep -qx "$d" "$1/busy_devices" 2>/dev/null || { echo "$d"; return 0; }
  done
  return 1
}

start() {  # start <dir> <device> <job...>: the job in the background, holding <dir>'s lock (fd 9)
  local dir=$1 d=$2
  shift 2
  exec 9>> "$dir/.lock"
  if ! flock -n 9; then exec 9>&-; return 1; fi
  echo "$d" > "$dir/device"
  "$@" &
  exec 9>&-
  SCAN=  # the next start scans afresh
}

lane_job() {  # lane_job <key> <arm> <device>: capture.sh unless DONE, then analyze.py on each kept
              # block of a DONE lane, on the device's CPU slice
  local key=$1 arm=$2 d=$3 A=$ROOT/$1/$2 cpus blk trace=()
  cpus=$(cpus_of "$d")
  if [ "$(status "$A")" != DONE ]; then
    # Task 1: not part of the compile-cache key; saves Wan bf16 ~10 min of NKI tracing
    case "$key $arm" in wan2[12]_official\ bf16) trace=(DIFFLET_NKI_TRACE_CACHE=1) ;; esac
    env HF_HUB_CACHE="$(hub_of "$key" "$arm")" "${trace[@]}" "$CAPTURE" --key "$key" --arm "$arm" \
      --device "$d" --cpus "$cpus" --out "$ROOT" >> "$A/run.log" 2>&1
    log "$key $arm on device $d: $(status "$A")"
    if [ "$(status "$A")" = BUSY ]; then echo "$d" >> "$A/busy_devices"; fi
  fi
  if [ "$(status "$A")" = DONE ]; then
    for blk in "$A"/blk*/; do
      grep -q "^analyze exit" "$blk/analyze.log" 2>/dev/null && continue
      taskset -c "$cpus" "$PY" "$HERE/analyze.py" "$blk" --model "$key" > "$blk/analyze.log" 2>&1
      echo "analyze exit $?" >> "$blk/analyze.log"
      log "$key $arm ${blk%/}: $(tail -1 "$blk/analyze.log")"
    done
  fi
  rm -f "$A/device"
}

baseline_job() {  # baseline_job <device> <dir> <lanes dumping>: BASELINE_NEFF captured as a lane's
                  # block (capture_neff), executions 4 and 8 summarized into <dir>/result.json
  local d=$1 B=$2 n=$3 cpus e
  cpus=$(cpus_of "$d")
  rm -f "$B/STATUS"
  note "$B" "baseline of $BASELINE_NEFF on device $d, cpus $cpus, $n lanes dumping"
  cp "$BASELINE_NEFF" "$B/block.neff"
  if capture_neff "$d" "$B" "$B" "$cpus"; then
    if cores_busy "$B/capture.log"; then write_status "$B" BUSY
    elif [ "$CAPTURE_RC" -ne 0 ]; then write_status "$B" "FAILED capture exit $CAPTURE_RC"
    else
      for e in 4 8; do
        ( cd "$B" && taskset -c "$cpus" timeout -k 60 900 neuron-explorer view -n block.neff \
            -s "profile_rank_0_exec_$e.ntff" --output-format summary-json ) \
          > "$B/summary_exec$e.json" 2>> "$B/view.err"
      done
      if "$PY" - "$B" "$d" "$n" > "$B/result.json" 2>> "$B/view.err" <<'EOF'
import json, re, sys
from pathlib import Path
blk, device, dumping = sys.argv[1:]
def summary(e):
    v = next(iter(json.loads(Path(blk, f"summary_exec{e}.json").read_text()).values()))
    limits = [x for k, x in v.items() if re.fullmatch(r"throttle_avg_util_limit_nc\d+_percent", k)]
    return {"total_time_ms": v["total_time"] * 1e3,
            "avg_util_limit": sum(limits) / len(limits) if limits else None}
print(json.dumps({"device": int(device), "lanes_dumping": int(dumping), "exec4": summary(4),
                  "exec8": summary(8)}))
EOF
      then write_status "$B" DONE; log "baseline ${B##*/}: $(< "$B/result.json")"
      else write_status "$B" "FAILED summary"; fi
    fi
  fi
  if [ "$(status "$B")" = BUSY ]; then echo "$d" >> "$B/busy_devices"; fi
  rm -f "$B/device"
}

maybe_baseline() {  # the in-run baseline (ruling R17): once, on a free idle device, while >= 4
                    # lanes are dumping
  local B=$ROOT/baseline/inrun n=0 lane d
  [ -n "$BASELINE_NEFF" ] || return 0
  case $(status "$B") in DONE|FAILED*) return 0 ;; esac
  running "$B" && return 0
  for lane in "${LANES[@]}"; do dumping "$ROOT/${lane/://}" && n=$((n + 1)); done
  [ "$n" -ge 4 ] || return 0
  [ -n "$SCAN" ] || SCAN="$(scan) "
  d=$(pick "$B") || return 0
  log "baseline inrun on device $d, cpus $(cpus_of "$d"), while $n lanes dump"
  start "$B" "$d" baseline_job "$d" "$B" "$n"
}

disk_ok() {  # disk_ok: 60 GB or more free on / (the plan's gate)
  local gb
  gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  [ "${gb:-0}" -ge 60 ] && return 0
  log "disk gate: ${gb:-?} GB free on /, under 60: no lane starts"
  return 1
}

done_marker() {  # RUN_DONE: each lane's STATUS and its blocks' analyses, the baselines
  local lane A blk B
  for lane in "${LANES[@]}"; do
    A=$ROOT/${lane/://}
    echo "${lane/:/ }: $(status "$A" || true)$([ -f "$A/STATUS" ] || echo "not started")"
    for blk in "$A"/blk*/; do
      [ -f "$blk/analyze.log" ] && echo "  ${blk%/}: $(tail -1 "$blk/analyze.log")"
    done
  done
  for B in "$ROOT"/baseline/*/; do
    B=${B%/}
    echo "baseline ${B##*/}: $(status "$B")$([ -f "$B/STATUS" ] || echo "not run")" \
      "$(cat "$B/result.json" 2>/dev/null)"
  done
}

if [ "${1:-}" = baseline ]; then  # run_all.sh baseline <device> <name>: one lone capture
  mkdir -p "$ROOT/baseline/$3"
  baseline_job "$2" "$ROOT/baseline/$3" 0
  [ "$(status "$ROOT/baseline/$3")" = DONE ]
  exit
fi

mkdir -p "$ROOT/baseline/inrun"
for lane in "${LANES[@]}"; do mkdir -p "$ROOT/${lane/://}"; done
rm -f "$ROOT/RUN_DONE" "$ROOT"/*/*/busy_devices  # a new run tries every device again
log "run_all start, pid $$"
declare -A WARM=()
while :; do
  SCAN= started=
  for lane in "${LANES[@]}"; do
    key=${lane%:*} arm=${lane#*:} A=$ROOT/${lane/://}
    if running "$A" || finished "$A"; then continue; fi
    if [ "$(status "$A")" != DONE ]; then  # it captures: the gate, its files
      if ! disk_ok; then
        for l in "${LANES[@]}"; do
          l=$ROOT/${l/://}
          running "$l" || finished "$l" || [ "$(status "$l")" = DONE ] ||
            write_status "$l" "FAILED disk"
        done
        started=1  # not an end yet: a DONE lane may still need its analyses
        break
      fi
      if [ -z "${WARM[$lane]:-}" ]; then
        prewarm "$A" "$key" "$arm" || continue
        WARM[$lane]=1 SCAN=
      fi
    fi
    [ -n "$SCAN" ] || SCAN="$(scan) "
    d=$(pick "$A") || continue
    log "start $key $arm on device $d, cpus $(cpus_of "$d"); idle:$SCAN"
    start "$A" "$d" lane_job "$key" "$arm" "$d" && started=1
    maybe_baseline
  done
  maybe_baseline
  if [ -z "$started" ]; then
    busy=
    for x in "$ROOT"/*/*/; do running "$x" && busy=1 && break; done
    if [ -z "$busy" ]; then
      done_marker > "$ROOT/RUN_DONE.tmp" && mv "$ROOT/RUN_DONE.tmp" "$ROOT/RUN_DONE"
      log "run_all end"
      exit 0
    fi
  fi
  sleep "$POLL"
done
