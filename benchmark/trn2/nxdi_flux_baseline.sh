#!/usr/bin/env bash
# Native NxDI FLUX.1-dev tp4 baseline on trn2, measured with the campaign rules.
# compile (fresh workdir, timed) -> drop page cache -> cold generate (1 process)
# -> warm generate (next process) -> benchmark/trn2/flux_1_dev_nxdi.{json,md}.
# Activate the Neuron venv first; run detached (scripts/supervise.sh).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE="${DIFFLET_BENCH_DEVICE:-trn2}"
PY="${PYTHON_BIN:-python}"
WORKDIR="${NXDI_WORKDIR:-$HOME/.cache/nxdi_flux/compiled_tp4_1024}"   # separate from ~/.cache/difflet
LOGDIR="benchmark/${DIFFLET_BENCH_DEVICE}/logs/nxdi"
OUTDIR="benchmark/${DIFFLET_BENCH_DEVICE}"
mkdir -p "$LOGDIR"
ts() { date -u +%FT%TZ; }
now() { date +%s.%N; }

echo "[nxdi] $(ts) compile -> $WORKDIR"
if [[ -f "$LOGDIR/compile.log" ]] && grep -q "NXDI_RESULT" "$LOGDIR/compile.log" && [[ -d "$WORKDIR/transformer" ]]; then
  echo "[nxdi] compile already recorded, reusing $WORKDIR"
else
  rm -rf "$WORKDIR"
  "$PY" -m benchmark.nxdi_flux_baseline compile --workdir "$WORKDIR" >"$LOGDIR/compile.log" 2>&1 \
    || { echo "[nxdi] compile FAILED"; tail -20 "$LOGDIR/compile.log"; exit 1; }
fi
grep NXDI_RESULT "$LOGDIR/compile.log" | tail -1

echo "[nxdi] $(ts) dropping page cache, cold generate"
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || echo "[nxdi] WARNING drop_caches failed"
t0=$(now)
"$PY" -m benchmark.nxdi_flux_baseline generate --workdir "$WORKDIR" --out "$OUTDIR/flux_1_dev_nxdi_out.png" \
  >"$LOGDIR/cold.log" 2>&1 || { echo "[nxdi] cold generate FAILED"; tail -20 "$LOGDIR/cold.log"; exit 1; }
cold_wall=$(echo "$(now) - $t0" | bc)
echo "[nxdi] cold wall ${cold_wall}s"; grep NXDI_RESULT "$LOGDIR/cold.log" | tail -1 | cut -c1-300

echo "[nxdi] $(ts) warm generate"
t0=$(now)
"$PY" -m benchmark.nxdi_flux_baseline generate --workdir "$WORKDIR" --out "$OUTDIR/flux_1_dev_nxdi_out.png" \
  >"$LOGDIR/warm.log" 2>&1 || { echo "[nxdi] warm generate FAILED"; tail -20 "$LOGDIR/warm.log"; exit 1; }
warm_wall=$(echo "$(now) - $t0" | bc)
echo "[nxdi] warm wall ${warm_wall}s"; grep NXDI_RESULT "$LOGDIR/warm.log" | tail -1 | cut -c1-300

"$PY" -m benchmark.nxdi_flux_baseline record --compile "$LOGDIR/compile.log" --cold "$LOGDIR/cold.log" \
  --warm "$LOGDIR/warm.log" --cold-wall "$cold_wall" --warm-wall "$warm_wall"
echo "[nxdi] $(ts) ALL_DONE"
