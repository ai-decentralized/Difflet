#!/usr/bin/env bash
# Best-combination search job (2026-10-03): for one model, run each combo label
# (benchmark.models.COMBO_LABELS) serially on the device: compile (bench
# --compile-only; a manifest hit for the runtime-only TeaCache labels takes
# seconds), warm e2e (benchmark.warm_e2e: 1 discarded cache-warming generate +
# 3 measured, fresh process each) and the real-loop per-step plus 3 in-process
# resident generate walls (benchmark.step_realloop --generates 3: the model
# stays loaded, the serving steady state; flux resets its TeaCache controller
# per call, difflet/models/flux/pipeline.py). No true-cold
# run: compile and cold time are out of scope for this search. Each metric is
# skipped when the cell JSON already has it, so a restart resumes per metric.
# Meant to run under .claude/skills/difflet-device-verify/scripts/supervise.sh.
#
#   benchmark/trn2/run_combo.sh <done-marker> <model> <label> [label ...]
#
# Env: DIFFLET_VENV (the Neuron venv), DIFFLET_BENCH_DEVICE (default trn2combo),
#      DIFFLET_MIN_FREE_GB (default 200), DIFFLET_WARM_ITERS (default 3).
set -u
MARKER="${1:?usage: run_combo.sh <done-marker> <model> <label> [label ...]}"
MODEL="${2:?model slug}"
shift 2
LABELS=("$@")
[[ ${#LABELS[@]} -gt 0 ]] || { echo "no labels" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export DIFFLET_VENV="${DIFFLET_VENV:-$ROOT/.venv}"
export PATH="$DIFFLET_VENV/bin:$PATH"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE="${DIFFLET_BENCH_DEVICE:-trn2combo}"
PY="$DIFFLET_VENV/bin/python"
MIN_FREE_GB="${DIFFLET_MIN_FREE_GB:-200}"
WARM_ITERS="${DIFFLET_WARM_ITERS:-3}"
ts() { date -u +%FT%TZ; }

need() {  # need <label> -> prints the missing metrics (compile warm step), space-separated
  "$PY" - "$MODEL" "$1" "$WARM_ITERS" <<'EOF'
import json, sys
from pathlib import Path
from benchmark.models import json_path, resolve
slug, label, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
p = Path(json_path(resolve(slug, label).config_slug))
d = json.loads(p.read_text()) if p.exists() else {}
miss = []
if not d.get("compile_seconds"):
    miss.append("compile")
if int((d.get("e2e_warm") or {}).get("n") or 0) < iters:
    miss.append("warm")
if not (d.get("step_latency") or {}).get("mean"):
    miss.append("step")
print(" ".join(miss))
EOF
}
run_step() {  # run_step <label> <name> <log> <cmd...>
  local label="$1" name="$2" log="$3"; shift 3
  echo "[combo] $(ts) $MODEL/$label $name start -> $log"
  local t0=$SECONDS
  if "$@" >"$log" 2>&1; then
    echo "[combo] $(ts) $MODEL/$label $name done in $((SECONDS - t0))s"
    return 0
  fi
  echo "[combo] $(ts) $MODEL/$label $name FAILED after $((SECONDS - t0))s; tail:"
  tail -n 15 "$log" | sed 's/^/    /'
  return 1
}

fail=0
for label in "${LABELS[@]}"; do
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if [[ -n "$free_gb" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
    echo "[combo] $(ts) LOW_DISK free=${free_gb}G < ${MIN_FREE_GB}G -- stopping; nothing is deleted without approval"
    fail=1; break
  fi
  logdir="benchmark/$DIFFLET_BENCH_DEVICE/logs/$label"
  mkdir -p "$logdir"
  miss="$(need "$label")"
  if [[ -z "$miss" ]]; then
    echo "[combo] $(ts) $MODEL/$label already complete"; continue
  fi
  ok=1
  if [[ " $miss " == *" compile "* ]]; then
    run_step "$label" compile "$logdir/${MODEL}_compile.log" \
      "$PY" -m benchmark.bench --model "$MODEL" --config "$label" --skip-download --compile-only || ok=0
  fi
  if [[ $ok == 1 && " $miss " == *" warm "* ]]; then
    run_step "$label" warm_e2e "$logdir/${MODEL}_warm.log" \
      "$PY" -m benchmark.warm_e2e --model "$MODEL" --config "$label" \
        --warmups 1 --iters "$WARM_ITERS" || ok=0
  fi
  if [[ $ok == 1 && " $miss " == *" step "* ]]; then
    run_step "$label" step_realloop "$logdir/${MODEL}_realloop.log" \
      "$PY" -m benchmark.step_realloop --model "$MODEL" --config "$label" \
        --generates "$WARM_ITERS" || ok=0
  fi
  if [[ $ok == 1 && -z "$(need "$label")" ]]; then
    echo "[combo] $(ts) $MODEL/$label CELL_COMPLETE"
  else
    echo "[combo] $(ts) $MODEL/$label CELL_INCOMPLETE (missing: $(need "$label"))"
    fail=1
  fi
done
echo "[combo] $(ts) ALL_DONE model=$MODEL fail=$fail" | tee "$MARKER"
exit "$fail"
