#!/usr/bin/env bash
# trn2 parallel-topology campaign driver: ONE feature (parallel-config label)
# across the campaign models, serially on the device.
#
#   benchmark/trn2/drive_parallel.sh <tp4|tp2cp2|tp4sp|tp2cfg> [model ...]
#
# Per model: by-design N/A cells (benchmark.models.UNSUPPORTED) are recorded
# with benchmark.mark_na; otherwise compile (benchmark.bench --compile-only),
# true cold + warm e2e (benchmark.cold_warm_e2e) and the H100-consistent
# per-step (benchmark.step_realloop) run in that order, each skipped when the
# cell's JSON already holds that metric (benchmark.cell_status), so a killed
# run resumes at step granularity and never re-records a manifest-hit
# "compile" over a real one. A failed step is logged and the driver moves on
# to the next model. Run it detached: scripts/supervise.sh drive_<label> <job>.
#
# Env: PYTHON_BIN (default: python on PATH -- activate the Neuron venv first),
#      DIFFLET_BENCH_DEVICE (default trn2), DIFFLET_MIN_FREE_GB (default 200).
set -u
LABEL="${1:?usage: drive_parallel.sh <label> [model ...]}"
shift
MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  # cheap first, Wan (VAE-dominated compile) last
  MODELS=(flux_1_dev qwen_image ltx_2 hunyuan_video wan_2_1)
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export DIFFLET_BENCH_DEVICE="${DIFFLET_BENCH_DEVICE:-trn2}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PYTHON_BIN:-python}"
MIN_FREE_GB="${DIFFLET_MIN_FREE_GB:-200}"
LOGDIR="benchmark/${DIFFLET_BENCH_DEVICE}/logs/${LABEL}"
mkdir -p "$LOGDIR"

ts() { date -u +%FT%TZ; }
disk() {
  df -h / | tail -1
  du -sh "$HOME/.cache/difflet" "$HOME/.cache/huggingface" 2>/dev/null
  local free_gb
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if [[ -n "$free_gb" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
    echo "[drive] LOW_DISK free=${free_gb}G < ${MIN_FREE_GB}G -- stopping before the next cell; nothing is deleted without approval"
    return 1
  fi
  return 0
}
is_unsupported() {
  "$PY" -c "import sys; from benchmark.models import UNSUPPORTED; sys.exit(0 if ('$1', '$2') in UNSUPPORTED else 1)"
}
missing() {   # prints the cell_status missing list (empty when complete)
  "$PY" -m benchmark.cell_status --model "$1" --config "$2" 2>/dev/null | sed -n 's/.*missing: //p'
}
run_step() {  # run_step <model> <step-name> <log> <cmd...>
  local m="$1" name="$2" log="$3"; shift 3
  echo "[drive] $(ts) $m/$LABEL $name start -> $log"
  local t0=$SECONDS
  if "$@" >"$log" 2>&1; then
    echo "[drive] $(ts) $m/$LABEL $name done in $((SECONDS - t0))s"
    return 0
  fi
  echo "[drive] $(ts) $m/$LABEL $name FAILED (exit $?) after $((SECONDS - t0))s; tail:"
  tail -n 15 "$log" | sed 's/^/    /'
  return 1
}

echo "[drive] === feature $LABEL on $DIFFLET_BENCH_DEVICE: ${MODELS[*]} ==="
"$PY" -c "from benchmark.models import NXD_VENV; print('[drive] venv', NXD_VENV)"
fail=0
for m in "${MODELS[@]}"; do
  if is_unsupported "$m" "$LABEL"; then
    "$PY" -m benchmark.mark_na --model "$m" --config "$LABEL" && echo "[drive] $m/$LABEL N/A (recorded)"
    continue
  fi
  miss="$(missing "$m" "$LABEL")"
  if [[ -z "$miss" ]] && "$PY" -m benchmark.cell_status --model "$m" --config "$LABEL" >/dev/null 2>&1; then
    echo "[drive] $m/$LABEL already complete, skipping"
    continue
  fi
  echo "[drive] === $m/$LABEL (missing: ${miss:-result file}) ==="
  disk || { fail=1; break; }
  ok=1
  if [[ "$miss" == *"compile_seconds"* || "$miss" == *"result file"* ]]; then
    run_step "$m" compile "$LOGDIR/${m}_bench.log" \
      "$PY" -m benchmark.bench --model "$m" --config "$LABEL" --skip-download --compile-only || ok=0
  else
    echo "[drive] $m/$LABEL compile already recorded, skipping"
  fi
  if [[ $ok == 1 ]]; then
    miss="$(missing "$m" "$LABEL")"
    if [[ "$miss" == *"e2e_cold"* || "$miss" == *"e2e_warm"* ]]; then
      run_step "$m" cold_warm_e2e "$LOGDIR/${m}_cold_warm.log" \
        "$PY" -m benchmark.cold_warm_e2e --model "$m" --config "$LABEL" || ok=0
    else
      echo "[drive] $m/$LABEL cold/warm e2e already recorded, skipping"
    fi
  fi
  if [[ $ok == 1 ]]; then
    miss="$(missing "$m" "$LABEL")"
    if [[ "$miss" == *"step_latency"* ]]; then
      run_step "$m" step_realloop "$LOGDIR/${m}_realloop.log" \
        "$PY" -m benchmark.step_realloop --model "$m" --config "$LABEL" || ok=0
    else
      echo "[drive] $m/$LABEL per-step already recorded, skipping"
    fi
  fi
  if [[ $ok == 1 ]] && "$PY" -m benchmark.cell_status --model "$m" --config "$LABEL"; then
    echo "[drive] $(ts) $m/$LABEL CELL_COMPLETE"
  else
    echo "[drive] $(ts) $m/$LABEL CELL_INCOMPLETE ($("$PY" -m benchmark.cell_status --model "$m" --config "$LABEL" 2>&1 | tail -1))"
    fail=1
  fi
done
"$PY" -m benchmark.cell_status --summary --config "$LABEL"
echo "[drive] ALL_DONE label=$LABEL fail=$fail"
exit "$fail"
