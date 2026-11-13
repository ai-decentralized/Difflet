#!/usr/bin/env bash
# One DiT block's device profile per model and arm (docs/plans/2026-11-13-dit-block-profile.md,
# Task 1), on one lane: one idle device and one CPU set. Into <root>/<key>/<arm>/:
#   1. dump: a 1-step `generate` (2 steps where the scheduler cannot take 1, see key_for) under
#      `neuron-explorer inspect` writes every rank's NEFFs and system profile under dump/, and
#      exits by itself (dump.log ends "dump exit <rc>");
#   2. candidates: the NEFFs over 1 MB whose sha256 is on every rank of rank 0's TP group and
#      whose debug info names model code and no VAE or text encoder (model_code), as many as the
#      key's block NEFFs, else FAILED candidates; the roles no shared NEFF fills matched per rank
#      when every rank compiled its own (per_rank, ruling R21); with a second TP group (HV1.5's
#      CFG halves, ranks 2-3) its NEFFs as well when their sha256 differ;
#   3. rank 0's system profile (system.json) and its step summary (system_summary.json,
#      system_profile.py);
#   4. per candidate, blk<i>/: block.neff (rank<r>.neff when every rank has its own) replayed
#      with its collectives as one worker per rank (`capture --num-exec=8 --profile-nth-exec=4`:
#      executions 4 and 8 profiled; again without --enable-dge-notifs after a notification queue
#      overflow, ruling R22), exported as summary.json (execution 4), summary_exec8.json and
#      pq/trace/*.parquet (rank 0's execution 4), then confirmed against signatures.json: a NEFF
#      whose instructions come mostly from a VAE or a text encoder, or that matches none of the
#      key's block kinds, fails the lane once every candidate is captured and checked;
#      source.json = {neff_name, sha256, ranks: [the NEFF's path in each rank's dump], kind,
#      per_rank, dge_notifs}.
# require_idle (lib.sh) runs before every device command (the dump, each capture); a capture also
# holds the host-wide capture lock (lib.sh's capture_neff, ruling R17). The lane ends
# in STATUS: DONE, FAILED <reason> (exit 1) or BUSY (exit 0, nothing more run on the device). A
# rerun skips what is done: the dump once dump.log says "dump exit 0", a block once its
# summary.json exists (written last); a dump redone removes the blocks and system profile first.
#
# usage: capture.sh --key <key> --arm <bf16|mxfp8> --device <d> --cpus <taskset list> --out <root>
#        capture.sh --key <key> --arm <bf16|mxfp8> --out <root> --candidates
#   keys: the plan's configuration table, flags from scripts/quant_bench/bench.sh's args_for.
#   --candidates: step 2 alone on a kept dump, its selection on stdout; no device, no file.
# env: PY (the native-fp8 venv's python), SIGNATURES (default: signatures.json here); inherited
#      as given: HF_HUB_CACHE, DIFFLET_NKI_TRACE_CACHE; CAPTURE_LOCK (lib.sh).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
. "$HERE/lib.sh"
PY=${PY:-$HOME/venvs/native-fp8/bin/python}
SIGNATURES=${SIGNATURES:-$HERE/signatures.json}
# neuronx-cc for a NEFF cache miss (bench.sh); neuron-explorer and fuser from PATH, the SDK's
# neuron-explorer last
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:$PATH:/opt/aws/neuron/bin
export HF_HUB_OFFLINE=1
# bench.sh's: shell cache locations would win over --cache-dir, thread counts change
# HunyuanVideo-1.5's host encode, and no arm here overrides the MXFP8 targets
unset DIFFLET_COMPILE_CACHE TORCH_NEURONX_NEFF_CACHE_DIR TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR \
  TORCH_NEURONX_HLO_CACHE_DIR TORCHINDUCTOR_CACHE_DIR DIFFLET_NEURON_WEIGHT_STORE \
  DIFFLET_SHARED_WEIGHTS_DIR OMP_NUM_THREADS MKL_NUM_THREADS DIFFLET_QUANT_TARGETS
DUMP_LIMIT=5400 # s; a cold compile inside the dump (HV1 544x960x129: ~42 min) fits
# source files of a VAE or a text encoder; of a DiT (difflet's model code, diffusers' transformers)
NOT_A_BLOCK='/vae/|/autoencoders/|text_encoder|/(u?mt5|t5|clip)/|/transformers/models/'
MODEL_CODE='/difflet/models/|/diffusers/models/transformers/'

WAN21_NEG='Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards'
WAN22_NEG='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

key_for() {  # key_for <key>: M, P, PROMPT, NEG and EXT as bench.sh's args_for (hunyuan_video_L:
             # hunyuan_video at evidence 7f's shape), with --steps STEPS; CACHE (ruling R7),
             # BLOCKS (block NEFFs per arm), TP (ranks per TP group), DGE (captures with
             # --enable-dge-notifs)
  NEG= EXT=mp4 PROMPT="a cat walking through a garden" TP=4 BLOCKS=1 STEPS=1 DGE=1
  CACHE=$HOME/difflet-cache-video
  case $1 in
    flux) M=(--model-id black-forest-labs/FLUX.1-dev --height 1024 --width 1024 --tp-degree 4)
      P=(--seed 42); PROMPT="a cat sitting on a bench"; EXT=png; BLOCKS=2 ;;
    qwen_image) M=(--model-id Qwen/Qwen-Image --height 1024 --width 1024 --tp-degree 4)
      P=(--seed 42 --guidance-scale 7.5); PROMPT="a cat sitting on a bench"; EXT=png
      CACHE=$HOME/difflet-cache-fp8; STEPS=2 ;;  # shift_terminal: below
    # Qwen-Image's and LTX-2's schedulers stretch the sigmas to shift_terminal (0.02, 0.1): one
    # step's single sigma divides 0 by 0, the timestep is NaN, and the scheduler's step finds no
    # index for it after the DiT forward (2026-11-13)
    ltx_2) M=(--model-id Lightricks/LTX-2 --height 512 --width 768 --num-frames 121 --tp-degree 4)
      P=(--seed 42); STEPS=2 ;;
    wan21_official) M=(--model-id Wan-AI/Wan2.1-T2V-14B-Diffusers
                       --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 --height 480
                       --width 832 --num-frames 81 --tp-degree 4)
      P=(--seed 42 --guidance-scale 5.0); NEG=$WAN21_NEG ;;
    wan22_official) M=(--model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers
                       --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7 --dual-expert
                       --height 480 --width 832 --num-frames 81 --tp-degree 4)
      P=(--seed 42 --guidance-scale 4.0 --guidance-scale-2 3.0); NEG=$WAN22_NEG ;;
    hunyuan_video_L) M=(--model-id hunyuanvideo-community/HunyuanVideo
                        --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773
                        --height 544 --width 960 --num-frames 129 --tp-degree 4)
      P=(--seed 42 --guidance-scale 6.0); BLOCKS=2
      # its 225-235 MB blocks overflow the notification queue with DGE notifications (pass 1,
      # both arms): captured without them from the start (ruling R23)
      DGE=
      PROMPT="a cinematic shot of a red fox running through a snowy forest" ;;
    hv15_official) M=(--model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
                      --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf
                      --height 480 --width 848 --num-frames 121)
      P=(--seed 42); TP=2 ;;  # tp2 x CFG2
    *) echo "capture.sh: unknown key $1" >&2; exit 2 ;;
  esac
  P=(--steps "$STEPS" "${P[@]}")
  if [ -n "$NEG" ]; then P+=(--negative-prompt "$NEG"); fi
  # HEAVY: a cold compile of this shape takes 11-42 min; its dump holds a dump slot (ruling R20)
  case $1 in wan21_official|wan22_official|hunyuan_video_L|hv15_official) HEAVY=1 ;; *) HEAVY= ;; esac
}

KEY= ARM= DEV= CPUS= OUT= ONLY=
while [ $# -gt 0 ]; do
  case $1 in
    --key) KEY=$2; shift 2 ;;
    --arm) ARM=$2; shift 2 ;;
    --device) DEV=$2; shift 2 ;;
    --cpus) CPUS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --candidates) ONLY=1; shift ;;
    *) echo "capture.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
case $ARM in
  bf16) QF=() ;;
  mxfp8) QF=(--quant mxfp8) ;;
  *) ARM= ;;
esac
if [ -z "$KEY" ] || [ -z "$ARM" ] || [ -z "$OUT" ] || { [ -z "$ONLY" ] && { [ -z "$DEV" ] || [ -z "$CPUS" ]; }; }; then
  echo "usage: capture.sh --key <key> --arm <bf16|mxfp8> --device <d> --cpus <list> --out <root>" >&2
  echo "       capture.sh --key <key> --arm <bf16|mxfp8> --out <root> --candidates" >&2
  exit 2
fi
if [ "$KEY $ARM" = "hv15_official mxfp8" ]; then
  echo "capture.sh: hv15_official mxfp8 is not a configuration: HunyuanVideo-1.5 has no MXFP8" \
       "default (the plan's table runs it in bf16 only)" >&2
  exit 2
fi
key_for "$KEY"
OUT=$(realpath -m -- "$OUT")  # the dump runs from the checkout; the lane tag needs one spelling
A=$OUT/$KEY/$ARM
if [ -n "$ONLY" ]; then  # --candidates: the kept dump's block NEFFs on stdout; no device, no file
  grep -qx "dump exit 0" "$A/dump.log" 2>/dev/null || { echo "capture.sh: no finished dump in $A" >&2; exit 2; }
  note() { shift; echo "$*" >&2; }
  fail() { echo "FAILED $*" >&2; exit 1; }
else
  mkdir -p "$A"
  rm -f "$A/STATUS"
  export NEURON_RT_VISIBLE_CORES=$((4 * DEV))-$((4 * DEV + 3))
  note "$A" "lane $KEY $ARM: device $DEV, cores $NEURON_RT_VISIBLE_CORES, cpus $CPUS"
  fail() { write_status "$A" "FAILED $*"; exit 1; }
fi
busy() { write_status "$A" BUSY; exit 0; }
on_device() {  # on_device <cmd...>: the lane's tag, its CPU set
  env DIFFLET_PROF_LANE="$A" taskset -c "$CPUS" "$@"
}
view() {  # view <ntff> <args...>: blk $B's NTFF read on the lane's CPUs (no device: no require_idle);
          # 2700 s (ruling R8): HV1.5's 2.17 GB NTFFs took 481-555 s per view, longer blocks more
  local ntff=$1; shift
  ( cd "$B" && taskset -c "$CPUS" timeout -k 60 2700 neuron-explorer view -n block.neff -s "$ntff" \
      "$@" ) \
    2>> "$B/view.err"
}

# 1. dump
if ! grep -qx "dump exit 0" "$A/dump.log" 2>/dev/null; then
  if [ -n "$HEAVY" ]; then dump_slot "$A"; fi  # held for the dump only, not the captures
  require_idle "$DEV" "$A" || exit 0
  # a new dump may compile other NEFFs: the old one's blocks and system profile go with it
  rm -rf "$A/dump" "$A/work" "$A"/blk* "$A/system.json" "$A/system_summary.json"
  note "$A" "dump start"
  ( cd "$REPO" && export PYTHONPATH="$REPO" NEURON_RT_INSPECT_DEVICE_PROFILE=1 \
      NEURON_RT_INSPECT_DEVICE_PROFILE_NUM_WARMUP=2 && on_device \
      timeout -s INT -k 90 "$DUMP_LIMIT" neuron-explorer inspect -o "$A/dump" \
      "$PY" -m difflet.cli.main generate --backend neuron "${M[@]}" "${QF[@]}" --cache-dir "$CACHE" \
      "${P[@]}" --prompt "$PROMPT" --output "$A/out.$EXT" --work-dir "$A/work" ) > "$A/dump.log" 2>&1
  rc=$?
  if [ -n "$HEAVY" ]; then flock -u 7; exec 7>&-; fi  # a leftover child must not keep the slot
  echo "dump exit $rc" >> "$A/dump.log"
  note "$A" "dump end, exit $rc"
  if cores_busy "$A/dump.log"; then busy; fi
  case $rc in
    0) ;;
    124|137) fail "timeout dump" ;;
    *) fail "dump exit $rc" ;;
  esac
fi

# 2. candidates; ranks are the dump's processes in pid order (torchrun starts them in rank
# order; a later process, as Wan's VAE decoder, comes after them). system_profile.py checks
# that the first one is rank 0.
mapfile -t RANKS < <(cd "$A/dump" && ls -d i-*_pid_* 2>/dev/null | sort -t_ -k3 -n)
if [ ${#RANKS[@]} -lt $WORLD ]; then fail "candidates: ${#RANKS[@]} process dumps, world $WORLD"; fi
model_code() {  # model_code <neff>: its debug info names model code and no VAE or text encoder
                # (FLUX's VAE pieces run on every rank too; some name no source file at all)
  local srcs
  # shortcut: a NEFF is a 1024-byte header and a tar.gz (neuronx-cc 2.28); upgrade if that moves
  srcs=$(tail -c +1025 "$1" | tar -xzOf - --wildcards '*.dbg' 2>/dev/null |
    grep -ao '/[A-Za-z0-9_./+-]*\.py' | sort -u)
  grep -qE "$MODEL_CODE" <<< "$srcs" && ! grep -qE "$NOT_A_BLOCK" <<< "$srcs"
}
candidates() {  # candidates <first rank>: "<size> <sha256> same <its path on each rank>" for each
                # NEFF over 1 MB of model code on every rank of the TP group from <first rank>,
                # largest first; the others go to the lane log
  local r line
  for ((r = $1; r < $1 + TP; r++)); do
    find "$A/dump/${RANKS[$r]}" -name '*.neff' -size +1M -printf '%s %p\n' |
      while read -r size path; do echo "$r $size $(sha256sum < "$path" | cut -c1-64) $path"; done
  done | awk -v tp="$TP" '!seen[$1, $3]++ { n[$3]++; s[$3] = $2; p[$3] = p[$3] " " $4 }
                          END { for (h in n) if (n[h] == tp) print s[h], h, "same" p[h] }' |
    sort -rn | while read -r line; do
      if model_code "$(cut -d' ' -f4 <<< "$line")"; then echo "$line"
      else note "$A" "on every rank, not model code: $line"; fi
    done
}
per_rank() {  # per_rank <n> <sha256 already chosen...>: "<size> <sha256> rank <path on rank 0..
              # WORLD-1>" (rank 0's size and sha256) for n roles whose NEFF differs between ranks
              # (ruling R21): on each rank its NEFFs over 1 MB of model code not chosen yet, largest
              # first, the i-th of every rank one role, within 5% of rank 0's size; else a line
              # "MISMATCH <reason>"
  local n=$1 r k tmp
  shift
  tmp=$(mktemp -d)
  for ((r = 0; r < WORLD; r++)); do
    find "$A/dump/${RANKS[$r]}" -name '*.neff' -size +1M -printf '%s %p\n' |
      while read -r size path; do
        sha=$(sha256sum < "$path" | cut -c1-64)
        case " $* " in *" $sha "*) continue ;; esac
        if model_code "$path"; then echo "$size $sha $path"; fi
      done | sort -rn > "$tmp/$r"
    k=$(wc -l < "$tmp/$r")
    if [ "$k" -ne "$n" ]; then
      echo "MISMATCH rank $r holds $k NEFFs over 1 MB of model code of its own, the table leaves $n"
      rm -rf "$tmp"
      return
    fi
  done
  (cd "$tmp" && paste -d' ' $(seq 0 $((WORLD - 1)))) | awk '{
    line = $1 " " $2 " rank"; sizes = ""
    for (i = 1; i <= NF; i += 3) { line = line " " $(i + 2); sizes = sizes " " $i
                                   if ($i > 1.05 * $1 || $i < 0.95 * $1) bad = 1 }
    if (bad) { print "MISMATCH role " NR " sizes" sizes " differ by over 5%"; exit }
    print line }'
  rm -rf "$tmp"
}
mapfile -t SEL < <(candidates 0)
if [ ${#SEL[@]} -lt "$BLOCKS" ]; then  # the rest differ between ranks (ruling R21)
  mapfile -t PR < <(per_rank $((BLOCKS - ${#SEL[@]})) $(printf '%s\n' "${SEL[@]}" | cut -d' ' -f2))
  mismatch=$(printf '%s\n' "${PR[@]}" | grep -m1 '^MISMATCH ')
  if [ -n "$mismatch" ]; then fail "candidates: ${mismatch#MISMATCH }"; fi
  mapfile -t SEL < <(printf '%s\n' "${SEL[@]}" "${PR[@]}" | sed '/^$/d' | sort -rn)
fi
if [ ${#SEL[@]} -ne "$BLOCKS" ]; then
  fail "candidates: ranks 0-$((TP - 1)) share ${#SEL[@]} NEFFs over 1 MB of model code," \
    "the table says $BLOCKS"
fi
GROUP0=${#SEL[@]}
if [ "$TP" -lt $WORLD ] && ! printf '%s\n' "${SEL[@]}" | cut -d' ' -f3 | grep -qx rank; then
  # the next TP group (ruling R3; a per-rank replay already gives every worker its own NEFF)
  mapfile -t C1 < <(candidates "$TP")
  if [ ${#C1[@]} -ne "$BLOCKS" ]; then
    fail "candidates: ranks $TP-$((2 * TP - 1)) share ${#C1[@]} NEFFs over 1 MB of model code," \
      "the table says $BLOCKS"
  fi
  for c in "${C1[@]}"; do
    printf '%s\n' "${SEL[@]}" | cut -d' ' -f2 | grep -qx "$(cut -d' ' -f2 <<< "$c")" || SEL+=("$c")
  done
fi
if [ -n "$ONLY" ]; then printf '%s\n' "${SEL[@]}"; exit 0; fi
printf '%s\n' "${SEL[@]}" | while read -r line; do note "$A" "candidate $line"; done

# 3. rank 0's system profile and its step
R0=$A/dump/${RANKS[0]}
if [ ! -s "$A/system_summary.json" ]; then
  # a full ring buffer drops events (HV1.5's VAE process, 2026-11-13): rank 0's step counts would
  # come out short; rerun the dump (remove dump.log) with NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC
  if grep -qE " ${RANKS[0]##*_pid_}:[0-9]+ .*System profile events were dropped" "$A/dump.log"; then
    fail "system profile: rank 0 dropped events (dump.log)"
  fi
  note "$A" "system profile start"
  taskset -c "$CPUS" timeout -k 60 900 neuron-explorer view -d "$R0" --output-format json \
    --output-file "$A/system.json" > "$A/system_view.log" 2>&1 || fail "system view exit $?"
  names=()
  for c in "${SEL[@]:0:$GROUP0}"; do names+=("$(basename "$(cut -d' ' -f4 <<< "$c")")"); done
  # the dump's own step count and step intervals check the loop window (ruling R9)
  if ! "$PY" "$HERE/system_profile.py" "$A/system.json" "$R0" "$STEPS" "$A/dump.log" "${names[@]}" \
      > "$A/system_summary.json.tmp" 2> "$A/system_profile.err"; then
    fail "system profile: $(tail -1 "$A/system_profile.err")"
  fi
  mv "$A/system_summary.json.tmp" "$A/system_summary.json"
  note "$A" "system profile end"
fi

# 4. each block: capture, export, confirm
i=0 rejected=()
for c in "${SEL[@]}"; do
  i=$((i + 1))
  B=$A/blk$i
  read -r _ sha mode paths <<< "$c"
  if [ ! -f "$B/summary.json" ]; then
    rm -rf "$B"
    mkdir -p "$B"
    cp "${paths%% *}" "$B/block.neff"
    if [ "$mode" = rank ]; then  # rank<r>.neff: capture_neff replays one worker per rank (R21)
      r=0
      for p in $paths; do cp "$p" "$B/rank$r.neff"; r=$((r + 1)); done
    fi
    if [ -z "$DGE" ]; then touch "$B/no_dge"; fi  # capture_neff leaves the flag out
    note "$A" "blk$i capture start (waits for the capture lock)"
    capture_neff "$DEV" "$A" "$B" "$CPUS" || exit 0  # require_idle wrote BUSY
    rc=$CAPTURE_RC
    note "$A" "blk$i capture end, exit $rc"
    if [ "$rc" -ne 0 ] && nq_overflow "$B/capture.log"; then
      # ruling R22: no runtime setting raises that queue (NEURON_RT_PROFILE_BUF_<type>_MB size
      # the profile buffers: DMA at 1024 MB overflowed the same); again without the DGE
      # notifications, the failed capture's partial NTFFs gone first
      note "$A" "blk$i notification queue overflow: capture again without --enable-dge-notifs"
      mv "$B/capture.log" "$B/capture_dge.log"
      rm -f "$B"/profile_rank_*.ntff
      touch "$B/no_dge"
      capture_neff "$DEV" "$A" "$B" "$CPUS" || exit 0
      rc=$CAPTURE_RC
      note "$A" "blk$i capture end, exit $rc"
    fi
    if cores_busy "$B/capture.log"; then busy; fi
    if [ $rc -ne 0 ]; then fail "capture blk$i exit $rc"; fi
    view profile_rank_0_exec_8.ntff --output-format summary-json > "$B/summary_exec8.json" \
      || fail "view blk$i exec 8 exit $?"
    view profile_rank_0_exec_4.ntff --output-format parquet --output-file "$B/pq/trace" \
      --ingest-only > "$B/parquet.log" || fail "view blk$i parquet exit $?"
    view profile_rank_0_exec_4.ntff --output-format summary-json > "$B/summary.json.tmp" \
      || fail "view blk$i exec 4 exit $?"
    mv "$B/summary.json.tmp" "$B/summary.json"
    note "$A" "blk$i views end"
  fi
  # shellcheck disable=SC2086 # paths: one per rank
  verdict=$("$PY" - "$B" "$KEY" "$SIGNATURES" "$NOT_A_BLOCK" "$sha" "$mode" $paths 2>&1 <<'EOF'
import hashlib, json, re, sys
from pathlib import Path
import pandas as pd
blk, key, signatures, not_a_block, sha, mode, *ranks = sys.argv[1:]
# use_threads=False: pyarrow's thread pool aborts ~3% of interpreter exits ("terminate called
# without an active exception", 4 of 150 runs; 0 of 400 without it), a block rejected for nothing
rows = pd.read_parquet(Path(blk, "pq/trace/Instruction.parquet"),
                       columns=["layer", "nki_source_location"], use_threads=False)
src = rows.nki_source_location.dropna().str.rsplit(":", n=1).str[0]
src = src[src != ""]
top = src.value_counts().idxmax() if len(src) else ""
if re.search(not_a_block, top):  # a VAE's or text encoder's rows come from its own code
    sys.exit(f"dominant source location {top}")
layers = rows.layer.dropna()
attention = rows.nki_source_location.str.contains("attention_cte", na=False).any()
kinds = json.loads(Path(signatures).read_text()).get(key, {})
match = [k for k, s in kinds.items() if all(layers.str.startswith(r).any() for r in s["require"])
         and (attention or not s.get("attention_kernel"))]
if not match:
    sys.exit(f"matches none of {key}'s signatures {sorted(kinds)}")
kind = max(match, key=lambda k: len(kinds[k]["require"]))
source = {"neff_name": Path(ranks[0]).name, "sha256": sha, "ranks": ranks, "kind": kind,
          "per_rank": mode == "rank", "dge_notifs": not Path(blk, "no_dge").exists()}
if mode == "rank":  # every rank its own NEFF (ruling R21); ranks[r] is rank r's
    source["sha256_by_rank"] = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ranks]
Path(blk, "source.json").write_text(json.dumps(source, indent=1) + "\n")
print(kind)
EOF
  ) || { rejected+=("blk$i: $(tail -1 <<< "$verdict")"); note "$A" "rejected ${rejected[-1]}"; continue; }
  note "$A" "blk$i kept: $verdict, $sha"
done
# every candidate captured and checked first: a new key's kinds all get their entries in one go
if [ ${#rejected[@]} -gt 0 ]; then fail "signature ${rejected[*]}"; fi
write_status "$A" DONE
