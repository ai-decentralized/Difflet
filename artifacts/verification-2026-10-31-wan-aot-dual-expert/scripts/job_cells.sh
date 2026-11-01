#!/usr/bin/env bash
# Wan 2.2 dual-expert (AOT, trainium) device campaign: every cell in order. A cell with a
# .result file is skipped, so a swept-and-restarted job resumes at the in-flight cell.
C=/home/ubuntu/campaigns/wan-dual-2026-10-31
source "$C/env.sh"
MAIN=$C/main-src
NM=/opt/aws/neuron/bin/neuron-monitor
CLI=(python -m difflet.cli.main)
COMMON=(--model-id "$MODEL" --revision "$REV" --height 480 --width 832 --num-frames 9
        --cache-dir "$R/cache")
GEN=(generate "${COMMON[@]}" --tp-degree 4 --steps 40 --guidance-scale 4.0 --seed 42
     --prompt "$PROMPT")

# cell NAME SRC TIMEOUT_S CMD...: run CMD against SRC's code, sampling neuron-monitor (HBM per
# core, host memory) throughout; record exit code, wall seconds and the peaks.
cell() {
  local name=$1 src=$2 limit=$3; shift 3
  local res=$C/logs/$name.result
  [ -f "$res" ] && return 0
  rm -f "$C/logs/$name.nm.jsonl"
  "$NM" > "$C/logs/$name.nm.jsonl" 2>/dev/null &
  local nm=$!
  local start rc
  start=$(date +%s)
  ( export SRC=$src; source "$C/env.sh"
    timeout "$limit" "$@" > "$C/logs/$name.log" 2>&1 )
  rc=$?
  kill "$nm" 2>/dev/null; wait "$nm" 2>/dev/null
  echo "$name exit=$rc secs=$(( $(date +%s) - start )) $(python "$C/hbm_peak.py" "$C/logs/$name.nm.jsonl")" > "$res"
  cat "$res"
}

ls "$R/cache" > "$C/logs/stage_dirs.before" 2>/dev/null

# --- 0. driver refusals: no stage may launch (the weights are local, so every check runs)
cell r_tp2cfg  "$WT" 300 "${CLI[@]}" compile "${COMMON[@]}" --tp-degree 2 --cfg-parallel --dual-expert
cell r_tp2cp2  "$WT" 300 "${CLI[@]}" compile "${COMMON[@]}" --tp-degree 2 --cp-degree 2 --cp-mode ring --dual-expert
cell r_dp2tp2  "$WT" 300 "${CLI[@]}" "${GEN[@]}" --tp-degree 2 --dp 2 --dual-expert --output "$C/out/r_dp2tp2.mp4"
cell r_gs2only "$WT" 300 "${CLI[@]}" "${GEN[@]}" --guidance-scale-2 3.0 --output "$C/out/r_gs2only.mp4"

# --- 1. main, single expert: the baseline the branch must reproduce bit for bit
cell main_compile "$MAIN" 7200 "${CLI[@]}" compile "${COMMON[@]}" --tp-degree 4
ls "$R/cache/wan_transformer" > "$C/logs/transformer_dirs.after_main" 2>/dev/null
for n in 1 2; do
  cell "main_gen$n" "$MAIN" 3600 "${CLI[@]}" "${GEN[@]}" --output "$C/out/main_gen$n.mp4" \
    --work-dir "$C/work/main_gen$n" --keep-work-dir
done

# --- 2. branch, single expert: same artifact (key unchanged), same latents
cell br_compile "$WT" 7200 "${CLI[@]}" compile "${COMMON[@]}" --tp-degree 4
ls "$R/cache/wan_transformer" > "$C/logs/transformer_dirs.after_branch_single" 2>/dev/null
cell br_gen1 "$WT" 3600 "${CLI[@]}" "${GEN[@]}" --output "$C/out/br_gen1.mp4" \
  --work-dir "$C/work/br_gen1" --keep-work-dir

# --- 3. branch, dual expert
cell dual_compile "$WT" 10800 "${CLI[@]}" compile "${COMMON[@]}" --tp-degree 4 --dual-expert
ls "$R/cache/wan_transformer" > "$C/logs/transformer_dirs.after_dual" 2>/dev/null
ls "$DIFFLET_SHARED_WEIGHTS_DIR" > "$C/logs/shared_weights.after_dual" 2>/dev/null
for n in 1 2 3; do
  cell "dual_gen$n" "$WT" 3600 "${CLI[@]}" "${GEN[@]}" --dual-expert --guidance-scale-2 3.0 \
    --output "$C/out/dual_gen$n.mp4" --work-dir "$C/work/dual_gen$n" --keep-work-dir
done
# --guidance-scale-2 equal to --guidance-scale must be the default (no --guidance-scale-2)
cell dual_gs2eq "$WT" 3600 "${CLI[@]}" "${GEN[@]}" --dual-expert --guidance-scale-2 4.0 \
  --output "$C/out/dual_gs2eq.mp4" --work-dir "$C/work/dual_gs2eq" --keep-work-dir
cell dual_nogs2 "$WT" 3600 "${CLI[@]}" "${GEN[@]}" --dual-expert \
  --output "$C/out/dual_nogs2.mp4" --work-dir "$C/work/dual_nogs2" --keep-work-dir
# fixed-cadence TeaCache across the expert boundary (a runtime option: same artifact)
cell dual_tc2 "$WT" 3600 "${CLI[@]}" "${GEN[@]}" --dual-expert --guidance-scale-2 3.0 \
  --teacache-cadence 2 --output "$C/out/dual_tc2.mp4" --work-dir "$C/work/dual_tc2" --keep-work-dir
# a two-request JSONL batch through the router at dp=1
cell dual_reqs "$WT" 3600 "${CLI[@]}" generate "${COMMON[@]}" --tp-degree 4 --dual-expert \
  --guidance-scale-2 3.0 --requests "$C/reqs.jsonl" --work-dir "$C/work/dual_reqs" --keep-work-dir

echo "ALL CELLS DONE $(date -u +%FT%TZ)" > "$C/logs/cells.done"
