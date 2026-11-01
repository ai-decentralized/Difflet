#!/usr/bin/env bash
# CPU fp32 diffusers dual-expert reference with one expert resident at a time (no device).
# First proves the swapped loop equals WanPipeline (bf16, 4 steps: 2 per expert), then runs
# fp32 at 40 steps. The fp32 run starts only if the check passes.
C=/home/ubuntu/campaigns/wan-dual-2026-10-31
source "$C/env.sh"
SNAP=~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/$REV
mkdir -p "$C/ref"
REF=(python tests/manual/wan_e2e_reference.py --model-dir "$SNAP" --out-dir "$C/ref"
     --prompt "$PROMPT" --guidance-scale 4.0 --guidance-scale-2 3.0 --seed 42)
run() {
  local name=$1; shift
  [ -f "$C/logs/$name.result" ] && return "$(sed -n 's/.*exit=\([0-9]*\).*/\1/p' "$C/logs/$name.result")"
  local start; start=$(date +%s)
  /usr/bin/time -v -o "$C/logs/$name.time" "${REF[@]}" "$@" > "$C/logs/$name.log" 2>&1
  local rc=$?
  echo "$name exit=$rc secs=$(( $(date +%s) - start )) max_rss_gb=$(awk -F': ' '/Maximum resident/ {printf "%.1f", $2/1048576}' "$C/logs/$name.time") $(grep -o 'check-swap.*' "$C/logs/$name.log" | tail -1)" > "$C/logs/$name.result"
  return $rc
}
if run check_swap --dtype bf16 --check-swap 4; then
  run ref_dual_fp32 --dtype fp32 --swap-experts --steps 40
fi
echo "FP32 REF DONE $(date -u +%FT%TZ)" > "$C/logs/fp32_ref.done"
