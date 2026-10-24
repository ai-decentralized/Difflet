#!/usr/bin/env bash
# Copy curated campaign evidence into the verify worktree (no latents, no NTFF, no NEFF).
set -eu
C=/home/ubuntu/campaign/bf16-reduce
V=/home/ubuntu/campaign/verify_wt
A=$V/artifacts/verification-2026-10-24-bf16-reduce
mkdir -p "$A" "$V/docs/verification"
cp "$C/report/evidence.md" "$V/docs/verification/bf16-reduce-2026-10-24-evidence.md"
cp "$C/report/qwen_cpu_fp32_bf16.jpg" "$C/report/ltx2_fp32_bf16.jpg" "$V/docs/verification/"
cp "$C/timings.txt" "$A/"
mkdir -p "$A/jobs" && cp "$C"/jobs/*.sh "$C"/jobs/*.py "$A/jobs/"
cd "$C/runs"
find . \( -name 'generate.log' -o -name 'steps.jsonl' -o -name 'summary.json' -o -name 'quality_*.json' \
  -o -name 'hlo_allreduce.*' -o -name 'excluded.txt' -o -name 'compile_*.log' -o -name 'reference.json' \
  -o -name 'table.json' -o -name 'rank0*.summary.json' -o -name 'capture_rank0.log' \) \
  -not -path './native*' -not -path './nxd_wan21*' -print0 | while IFS= read -r -d '' f; do
    mkdir -p "$A/runs/$(dirname "$f")"; cp "$f" "$A/runs/$f"
  done
for f in */*/warm1/out.png */*/warm1/out.mp4 cpu_ref/qwen_image/out.png cpu_ref/ltx_2/out.mp4; do
  [ -e "$f" ] || continue
  mkdir -p "$A/runs/$(dirname "$f")"; cp "$f" "$A/runs/$f"
done
du -sh "$A"
find "$A" -size +10M
