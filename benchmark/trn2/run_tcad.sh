#!/usr/bin/env bash
# tp4tcad campaign job: per model, wait for its download, run the prep
# (benchmark.tcad_prep: tp4 artifact + reference output, adaptive artifact,
# calibration) and then the cell (drive_parallel.sh tp4tcad). Serial on the
# device, resumable at every step; meant to run under
# .claude/skills/difflet-device-verify/scripts/supervise.sh, which restarts it
# when a swept task takes it down before the done marker.
#
#   benchmark/trn2/run_tcad.sh <done-marker> [model ...]
#
# Env: DIFFLET_VENV (the Neuron venv), DIFFLET_DL_LOG (download log to wait on;
#      unset = do not wait), DIFFLET_TCAD_PROMPTS (calibration prompts, default 3).
set -u
MARKER="${1:?usage: run_tcad.sh <done-marker> [model ...]}"
shift
MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(flux_1_dev qwen_image ltx_2 hunyuan_video wan_2_1)
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export DIFFLET_VENV="${DIFFLET_VENV:-$ROOT/.venv}"
export PATH="$DIFFLET_VENV/bin:$PATH"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE="${DIFFLET_BENCH_DEVICE:-trn2}"
LOGDIR="benchmark/${DIFFLET_BENCH_DEVICE}/logs/tp4tcad_prep"
mkdir -p "$LOGDIR"
ts() { date -u +%FT%TZ; }
fail=0
for m in "${MODELS[@]}"; do
  if [[ -n "${DIFFLET_DL_LOG:-}" ]]; then
    until grep -q "\[dl\] .* $m done" "$DIFFLET_DL_LOG" 2>/dev/null; do
      if grep -q "\[dl\] .* $m FAILED" "$DIFFLET_DL_LOG" 2>/dev/null; then
        echo "[tcad] $(ts) $m download FAILED; skipping"; fail=1; continue 2
      fi
      sleep 30
    done
  fi
  echo "[tcad] $(ts) === $m: prep ==="
  if python -m benchmark.tcad_prep --model "$m" --prompts "${DIFFLET_TCAD_PROMPTS:-3}" \
       > "$LOGDIR/${m}_prep.log" 2>&1; then
    echo "[tcad] $(ts) $m PREP_COMPLETE"
  else
    echo "[tcad] $(ts) $m PREP_FAILED (exit $?); tail:"; tail -n 20 "$LOGDIR/${m}_prep.log" | sed 's/^/    /'
    fail=1; continue
  fi
  echo "[tcad] $(ts) === $m: cell tp4tcad ==="
  if bash benchmark/trn2/drive_parallel.sh tp4tcad "$m"; then
    echo "[tcad] $(ts) $m CELL_DONE"
  else
    echo "[tcad] $(ts) $m CELL_FAILED"; fail=1
  fi
done
echo "[tcad] $(ts) ALL_DONE fail=$fail" | tee "$MARKER"
exit "$fail"
