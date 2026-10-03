#!/usr/bin/env bash
# Per-model FP8 PTQ device verification: bf16 and fp8 (W8A8, dynamic activations) through the
# benchmark harness (bench + true cold / warm e2e), then output comparisons against the
# bf16 output and the A4 fingerprints (store entries). Serialized on the device.
#
#   source .venv/bin/activate && export PYTHONPATH=$PWD
#   scripts/ptq_model_verify.sh <bf16-slug>            # e.g. flux_1_dev, qwen_image, ltx_2, hunyuan_video, wan_2_2
#   DRY=1 scripts/ptq_model_verify.sh <bf16-slug>      # print the commands only
#
# Results: benchmark/trn2/<slug>{,_fp8}.{json,md} (patched by the harness) and
# artifacts/verification-2026-10-02/ptq-all/<slug>/ (logs, compare JSONs, outputs, store listing).
set -uo pipefail
SLUG=${1:?usage: ptq_model_verify.sh <bf16-slug>}
DRY=${DRY:-0}
EVID=artifacts/verification-2026-10-02/ptq-all/$SLUG
RESULTS=benchmark/trn2
mkdir -p "$EVID/logs"

run() {  # run "<label>" cmd...
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  if [ "$DRY" = 1 ]; then echo "   $*"; return 0; fi
  "$@" 2>&1 | grep -vE 'Warning|warnings.warn' | tee "$EVID/logs/$label.log" | tail -4
  return "${PIPESTATUS[0]}"
}

if [ "$DRY" != 1 ]; then
  bash scripts/gate_idle.sh | tee "$EVID/gate.txt" | tail -1 | grep -q IDLE || { echo "host busy; see $EVID/gate.txt"; exit 2; }
fi
MODEL_ID=$(python -c "from benchmark.models import MATRIX; print(MATRIX['$SLUG'].model_id)")
REV=$(python -c "from benchmark.models import MATRIX; print(MATRIX['$SLUG'].revision or '')")
echo "model $MODEL_ID revision ${REV:-main}"

# SKIP_BF16=1 re-runs only the fp8 arm (after a device fix) against an already measured bf16 arm.
# ARMS overrides the arm list (default: bf16 + dynamic fp8), e.g. ARMS="_fp8_static" for the
# calibrated static-scale arm alone (its calibration JSON must exist, see benchmark/models.py).
ARMS=${ARMS:-"\"\" _fp8"}
eval "arm_list=($ARMS)"
for arm in "${arm_list[@]}"; do
  if [ -z "$arm" ] && [ "${SKIP_BF16:-0}" = 1 ]; then echo "=== bf16 arm skipped (SKIP_BF16=1)"; continue; fi
  s=$SLUG$arm
  if [ -n "$arm" ]; then
    # quantize (CPU, once per arm) with exactly the arm's quant flags, so a static arm
    # builds its calibrated checkpoint here and not lazily inside the compile stage
    QFLAGS=$(python -c "from benchmark.models import MATRIX; print(' '.join(MATRIX['$s'].quant_flags()))")
    run "quantize_$s" python -m difflet.cli.main quantize --model-id "$MODEL_ID" ${REV:+--revision "$REV"} $QFLAGS
    echo "QUANTIZE_${s}_RC=$?"
  fi
  run "bench_$s" python -m benchmark.bench --model "$s" --skip-download --iters 1
  echo "BENCH_${s}_RC=$?"
  run "cold_warm_$s" python -m benchmark.cold_warm_e2e --model "$s"
  echo "COLDWARM_${s}_RC=$?"
done
if [ "$DRY" = 1 ]; then echo "DRY_DONE $SLUG"; exit 0; fi

# Output comparisons: the harness writes <results>/<spec_slug>_out.<ext> per arm.
bf16_slug=$(python -c "from benchmark.models import MATRIX; from benchmark.adapters.trainium import spec_slug; print(spec_slug(MATRIX['$SLUG']))")
REF=$(ls "$RESULTS/${bf16_slug}_out".* 2>/dev/null | grep -vE '\.pt$' | head -1)
for arm in "${arm_list[@]}"; do
  [ -n "$arm" ] || continue
  arm_slug=$(python -c "from benchmark.models import MATRIX; from benchmark.adapters.trainium import spec_slug; print(spec_slug(MATRIX['$SLUG$arm']))")
  TEST=$(ls "$RESULTS/${arm_slug}_out".* 2>/dev/null | grep -vE '\.pt$' | head -1)
  if [ -n "$REF" ] && [ -n "$TEST" ]; then
    python scripts/ptq_compare_outputs.py --reference "$REF" --test "$TEST" --out "$EVID/compare${arm}_vs_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -2
    cp "$TEST" "$EVID/" 2>/dev/null
  else
    echo "compare${arm}: missing output (ref=$REF test=$TEST)"
  fi
done
[ -n "$REF" ] && cp "$REF" "$EVID/" 2>/dev/null
cp "$RESULTS/$SLUG".json "$RESULTS/$SLUG".md "$RESULTS/${SLUG}_fp8".json "$RESULTS/${SLUG}_fp8".md "$EVID/" 2>/dev/null
ls -la ~/.cache/difflet/_shared_weights 2>/dev/null | grep -i "$(echo "$MODEL_ID" | sed 's|/|--|')" > "$EVID/store_entries.txt"
du -sh ~/.cache/difflet/quantized/*/transformer*/ 2>/dev/null >> "$EVID/store_entries.txt"
echo "VERIFY_DONE $SLUG"
