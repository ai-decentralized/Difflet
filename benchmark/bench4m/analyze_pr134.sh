#!/usr/bin/env bash
# PR #134 verification analysis (CPU only; run after the device runs). Same tool, references and
# definitions as the 2026-10-26 four-model campaign (quality_part1.sh / quality_part2_hv_aot.sh):
# the CPU fp32 references with the AoT noise, 50 steps, seed 42, the fox prompt.
set -eu
D=/home/ubuntu/dv-2026-11-01-aot
B=/home/ubuntu/bench-2026-10-26-4m
PY=/home/ubuntu/venvs/difflet-neuron/bin/python
export PATH=/home/ubuntu/venvs/difflet-neuron/bin:$PATH
M=$D/scripts/quality_metrics.py
MAP=$D/scripts/frames_map.py
O=$D/analysis
mkdir -p "$O/mapped"
cd /tmp
"$PY" "$M" --selftest > "$O/selftest.txt" 2>&1

# Qwen-Image: each side vs the CPU fp32 reference, and after vs before.
QR=$B/qwen_image/cpu_ref/ref_aot_noise_50
for cell in after_tp4 before_tp4; do
  Q=$D/qwen_image/$cell/q1
  [ -f "$Q/final.pt" ] || { echo "skip qwen $cell (no q1)"; continue; }
  "$PY" "$M" --label "qwen_image AoT $cell q1 vs CPU fp32 ref_aot_noise_50, 50 steps" \
    --latents final "$QR/final.pt" "$Q/final.pt" \
    --ref-frames "$QR/out.pt" --cand-frames "$Q/out.png" --out "$O/qwen_${cell}_vs_ref.json"
done
if [ -f "$D/qwen_image/after_tp4/q1/final.pt" ] && [ -f "$D/qwen_image/before_tp4/q1/final.pt" ]; then
  "$PY" "$M" --label "qwen_image AoT after vs before (q1)" \
    --latents final "$D/qwen_image/before_tp4/q1/final.pt" "$D/qwen_image/after_tp4/q1/final.pt" \
    --ref-frames "$D/qwen_image/before_tp4/q1/out.png" --cand-frames "$D/qwen_image/after_tp4/q1/out.png" \
    --out "$O/qwen_after_vs_before.json"
fi

# HunyuanVideo 1.0: frames mapped as the reference's, each side vs the reference, after vs before.
HR=$B/hunyuan_video/cpu_ref/ref_aot_noise_50
: > "$O/hv_frames_map.jsonl"
for cell in after_tp4 before_tp4; do
  A=$D/hunyuan_video/$cell/q1
  [ -f "$A/out.pt" ] || { echo "skip hv $cell (no q1)"; continue; }
  "$PY" "$MAP" "$A/out.pt" "$O/mapped/hv_${cell}_q1.pt" >> "$O/hv_frames_map.jsonl"
  "$PY" "$M" --label "hunyuan_video AoT $cell q1 vs CPU fp32 ref_aot_noise_50, 50 steps" \
    --latents final "$HR/final.pt" "$A/final.pt" \
    --ref-frames "$HR/out.pt" --cand-frames "$O/mapped/hv_${cell}_q1.pt" --out "$O/hv_${cell}_vs_ref.json"
done
if [ -f "$O/mapped/hv_after_tp4_q1.pt" ] && [ -f "$O/mapped/hv_before_tp4_q1.pt" ]; then
  "$PY" "$M" --label "hunyuan_video AoT after vs before (q1)" \
    --latents final "$D/hunyuan_video/before_tp4/q1/final.pt" "$D/hunyuan_video/after_tp4/q1/final.pt" \
    --ref-frames "$O/mapped/hv_before_tp4_q1.pt" --cand-frames "$O/mapped/hv_after_tp4_q1.pt" \
    --out "$O/hv_after_vs_before.json"
fi
echo "analysis written to $O"
