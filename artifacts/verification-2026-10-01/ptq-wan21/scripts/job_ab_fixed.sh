#!/usr/bin/env bash
# Re-measure the fp8 arms after bug 5 (layer dtype) with the schema-bumped cache keys: dynamic, then weight-only.
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
COMMON=(--model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 --seed 42 \
  --runs 2 --drop-caches --only fp8 --skip-quantize)
bash /home/ubuntu/.claude/jobs/b5f130d0/tmp/gate_idle.sh | tee artifacts/verification-2026-10-01/ptq-wan21/gate_before_ab_fixed.txt | tail -1
echo "=== ab-fixed (dynamic) start $(date -u +%FT%TZ)"
python scripts/ptq_fp8_ab.py "${COMMON[@]}" --quant-act dynamic --out-dir "$PWD/artifacts/verification-2026-10-01/ptq-wan21/ab-fixed"
echo "AB_FIXED_DYN_RC=$?"
echo "=== ab-fixed (weight-only) start $(date -u +%FT%TZ)"
python scripts/ptq_fp8_ab.py "${COMMON[@]}" --quant-act none --out-dir "$PWD/artifacts/verification-2026-10-01/ptq-wan21/ab-wo-fixed"
echo "AB_FIXED_WO_RC=$?"
AB=$PWD/artifacts/verification-2026-10-01/ptq-wan21/ab
for d in ab-fixed ab-wo-fixed; do
  O=$PWD/artifacts/verification-2026-10-01/ptq-wan21/$d
  python scripts/ptq_compare_outputs.py --reference "$AB/bf16_run1.mp4" --test "$O/fp8_run1.mp4" \
    --latents-reference "$AB/work_bf16_run1/latents.pt" --latents-test "$O/work_fp8_run1/latents.pt" \
    --out "$O/compare_fp8_vs_bf16_run1.json" 2>&1 | grep -vE 'Warning|warn' | tail -3
done
echo "AB_FIXED_DONE"
