#!/usr/bin/env bash
# Per-engine device profile of the static-scale Wan 2.1 fp8 transformer NEFF (MODULE_caa9e95d…), tp4,
# same capture settings as the bf16 / fp8-dynamic profiles in profile/.
set -uo pipefail
W=/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
cd "$W"
source .venv/bin/activate
export PYTHONPATH=$PWD
E=$W/artifacts/verification-2026-10-03/fp8-dyn-step/profile
mkdir -p "$E"
bash scripts/gate_idle.sh | tee "$E/gate_static.txt" | tail -1 | grep -q IDLE || { echo "host busy"; exit 2; }
C=/var/tmp/neuron-compile-cache/neuronxcc-2.26.6360.0+6f180f47
name=fp8_static; mod=MODULE_caa9e95d65eaa97043ba+adc74e56
echo "=== capture $name $mod $(date -u +%FT%TZ)"
rm -rf "$E/$name"; mkdir -p "$E/$name"
( cd "$E/$name" && /opt/aws/neuron/bin/neuron-explorer capture -n "$C/$mod/model.neff" -s profile.ntff \
    -r 4 --collectives-worker-count 4 --collectives-profile-id 0 --num-exec 3 --profile-nth-exec 3 --ignore-exec-errors 2>&1 \
    | grep -vE 'Warning|warn' | tail -6 )
echo "CAPTURE_${name}_RC=$?"
ntff=$(ls "$E/$name"/*.ntff 2>/dev/null | head -1)
if [ -n "$ntff" ]; then
  echo "=== view $name $(date -u +%FT%TZ)"
  /opt/aws/neuron/bin/neuron-explorer view -n "$C/$mod/model.neff" -s "$ntff" --output-format summary-text > "$E/$name/summary_full.txt" 2> "$E/$name/view_text.err"
  echo "VIEW_TEXT_RC=$? bytes=$(wc -c < "$E/$name/summary_full.txt")"
  /opt/aws/neuron/bin/neuron-explorer view -n "$C/$mod/model.neff" -s "$ntff" --output-format summary-json --output-file "$E/$name/summary.json" > "$E/$name/view_json.out" 2>&1
  echo "VIEW_JSON_RC=$? bytes=$(wc -c < "$E/$name/summary.json" 2>/dev/null)"
  mkdir -p /home/ubuntu/ptq-profiles
  mv "$ntff" /home/ubuntu/ptq-profiles/wan21_fp8static_2_26_rank0.ntff && echo "ntff moved out of the repo"
  rm -f "$E/$name"/*.ntff
fi
echo "PROFILE_STATIC_DONE"
