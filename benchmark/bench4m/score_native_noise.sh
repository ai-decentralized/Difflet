#!/usr/bin/env bash
# Qwen-Image AoT on the native run's noise ("qn" runs) against ref_native_noise_50 (the CPU fp32
# reference the native fix's 32.9 dB was measured against), plus that noise's bf16 floor
# (diffusers' bf16 pipeline vs its fp32 pipeline). Missing qn runs are skipped.
set -u
D=/home/ubuntu/dv-2026-11-01-aot
B=/home/ubuntu/bench-2026-10-26-4m
R=$B/qwen_image/cpu_ref/ref_native_noise_50
R16=$B/qwen_image/cpu_ref/ref_native_noise_50_bf16
PY=/home/ubuntu/.claude/jobs/1c9be1ce/tmp/aotpy
M=$D/scripts/quality_metrics.py
O=$D/analysis
cd /home/ubuntu/Difflet/.claude/worktrees/aot-followups
"$PY" "$M" --label "qwen CPU bf16 pipeline vs CPU fp32 pipeline (native noise): the bf16 floor" \
  --latents final "$R/final.pt" "$R16/final.pt" --ref-frames "$R/out.pt" --cand-frames "$R16/out.pt" \
  --out "$O/qwen_bf16ref_vs_fp32ref_native_noise.json" > /dev/null 2>&1
outs=("$O/qwen_bf16ref_vs_fp32ref_native_noise.json")
for cell in after_tp4 before_tp4; do
  Q=$D/qwen_image/$cell/qn
  [ -f "$Q/final.pt" ] || continue
  "$PY" "$M" --label "qwen AoT $cell qn (native noise) vs CPU fp32 ref_native_noise_50" \
    --latents final "$R/final.pt" "$Q/final.pt" --ref-frames "$R/out.pt" --cand-frames "$Q/out.png" \
    --out "$O/qwen_${cell}_qn_vs_native_ref.json" > /dev/null 2>&1
  outs+=("$O/qwen_${cell}_qn_vs_native_ref.json")
done
outs+=("$B/quality_part1/qwen_native_vs_ref.json")
/usr/bin/python3 "$D/scripts/brief.py" "${outs[@]}"
