#!/usr/bin/env bash
# CPU bf16 diffusers references (no device): Wan 2.2 dual expert, then single expert for
# calibration; same seed, prompt, steps and guidance as the device cells.
C=/home/ubuntu/campaigns/wan-dual-2026-10-31
source "$C/env.sh"
SNAP=$(ls -d ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/$REV)
mkdir -p "$C/ref"
REF=(python tests/manual/wan_e2e_reference.py --model-dir "$SNAP" --out-dir "$C/ref" --dtype bf16
     --steps 40 --prompt "$PROMPT" --guidance-scale 4.0 --seed 42)
run() {
  local name=$1; shift
  [ -f "$C/logs/$name.result" ] && return 0
  local start; start=$(date +%s)
  /usr/bin/time -v -o "$C/logs/$name.time" "${REF[@]}" "$@" > "$C/logs/$name.log" 2>&1
  local rc=$?
  echo "$name exit=$rc secs=$(( $(date +%s) - start )) max_rss_gb=$(awk -F': ' '/Maximum resident/ {printf "%.1f", $2/1048576}' "$C/logs/$name.time")" > "$C/logs/$name.result"
}
run ref_dual --dual-expert --guidance-scale-2 3.0
run ref_single
echo "CPU REFS DONE $(date -u +%FT%TZ)" > "$C/logs/cpu_ref.done"
