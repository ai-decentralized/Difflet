#!/usr/bin/env bash
# Fixed-cadence TeaCache A/B on the warm tp4 cache: baseline vs --teacache-cadence 2.
#   REPO=... SHAPE_FLAGS="--height 480 --width 832 --num-frames 9" \
#     bash teacache_cadence_ab.sh <model-id> <steps> [out-ext png|mp4]
# SHAPE_FLAGS must match the compiled artifact's shape (CLI defaults are the
# model defaults, which may not be the warm cache you verified with) or the
# baseline run starts a multi-hour compile instead of a timed generate.
# Positive evidence required: the "[teacache] stats" line with skipped_steps > 0
# AND differing outputs; a faster wall-clock alone can be page-cache warmth.
set -uo pipefail
MODEL=$1; STEPS=$2; EXT=${3:-png}
REPO=${REPO:-$PWD}; OUT=${OUT:-/tmp/logs/cadence_ab}; mkdir -p "$OUT"
cd "$REPO" && source .venv/bin/activate
slug=$(echo "$MODEL" | tr '/' '_')
run() { local label=$1; shift; local t0=$(date +%s.%N)
  python -m difflet.cli.main generate --model-id "$MODEL" --tp-degree 4 ${SHAPE_FLAGS:-} --steps "$STEPS" --seed 42 "$@" \
    --prompt "a cat sitting on a bench" --output "$OUT/${slug}_${label}.$EXT" > "$OUT/${slug}_${label}.log" 2>&1
  echo "$label rc=$? wall=$(echo "$(date +%s.%N) $t0" | awk '{printf "%.1f", $1-$2}')s"; }
run baseline
run cadence2 --teacache-cadence 2
grep -h "\[teacache\]" "$OUT/${slug}_cadence2.log" || echo "NO [teacache] stats line — cadence did not engage"
cmp -s "$OUT/${slug}_baseline.$EXT" "$OUT/${slug}_cadence2.$EXT" && echo "outputs IDENTICAL — no-op" || echo "outputs differ (expected)"
echo "CADENCE_AB DONE" | tee "$OUT/${slug}_DONE"
