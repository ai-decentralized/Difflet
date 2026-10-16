#!/usr/bin/env bash
# The CPU gate for the CUDA-backend branch: import guard + unit suite with the
# CPU reference backend selected, on a host with no Neuron SDK and no GPU.
#
# Prints a one-line summary last, so it can be the test command of
# executing-plans' `task-done` (which records the last non-empty line).
#
# Usage: cpu-gate.sh [--unit-only] [extra pytest args...]
#
# Red tests are compared against known-red.txt next to this script: the exact
# test ids that were red on a non-Neuron host when the branch started (every
# one a missing Neuron module or a trainium-only model registration). Any red
# test NOT in that file fails the gate — that is a regression you caused. Known
# red tests that now pass are reported as progress; remove them from the file
# in the same commit that fixed them, so the file shrinks to empty by the end
# of Phase 1.
#
# CPU_GATE_IMPORTS_REQUIRED=1 makes a failing scripts/test_imports.sh fail the
# gate. It is 0 by default until plan task 1.7 lands; flip it in that commit.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$ROOT"

PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export PYTHON_BIN="$PY"
export DIFFLET_BACKEND=cpu
export PATH="$(dirname "$PY"):$PATH"

LOGDIR="${CPU_GATE_LOGDIR:-/tmp/logs}"
mkdir -p "$LOGDIR"
ts="$(date +%Y%m%d-%H%M%S)"
unit_only=0
if [ "${1:-}" = "--unit-only" ]; then unit_only=1; shift; fi

imports="skipped"
if [ "$unit_only" -eq 0 ]; then
  if ./scripts/test_imports.sh > "$LOGDIR/cpu-gate-imports-$ts.log" 2>&1; then
    imports="ok"
  else
    imports="FAIL"
    grep -E 'ModuleNotFoundError|Error' "$LOGDIR/cpu-gate-imports-$ts.log" | tail -n 2 >&2
  fi
fi

unit_log="$LOGDIR/cpu-gate-unit-$ts.log"
"$PY" -m pytest tests/unit -q --tb=short -p no:cacheprovider -p no:warnings \
  --continue-on-collection-errors "$@" > "$unit_log" 2>&1
summary="$(grep -E '^[0-9]+ (passed|failed)' "$unit_log" | tail -n 1 || true)"
[ -n "$summary" ] || summary="$(tail -n 1 "$unit_log")"

known="$HERE/known-red.txt"
[ -f "$known" ] || : > "$known"
red_now="$(grep -E '^(FAILED|ERROR) ' "$unit_log" | sed -E 's/^(FAILED|ERROR) //; s/ - .*//' | sort -u)"
unexpected="$(comm -23 <(printf '%s\n' "$red_now" | sed '/^$/d') <(sort -u "$known"))"
fixed="$(comm -13 <(printf '%s\n' "$red_now" | sed '/^$/d') <(sort -u "$known"))"
n_known="$(sed '/^$/d' "$known" | sort -u | wc -l | tr -d ' ')"
n_fixed="$(printf '%s\n' "$fixed" | sed '/^$/d' | wc -l | tr -d ' ')"

status=0
if [ -n "$unexpected" ]; then
  echo "REGRESSION: red tests not in known-red.txt:" >&2
  printf '%s\n' "$unexpected" | head -n 20 >&2
  status=1
fi
if [ "$imports" = "FAIL" ] && [ "${CPU_GATE_IMPORTS_REQUIRED:-0}" = "1" ]; then
  status=1
fi
if [ "$n_fixed" -gt 0 ]; then
  echo "known-red now passing ($n_fixed) — remove them from known-red.txt in this commit:" >&2
  printf '%s\n' "$fixed" | head -n 20 >&2
fi

echo "cpu-gate: imports=$imports unit=[$summary] known_red=$n_known fixed=$n_fixed regressions=$(printf '%s\n' "$unexpected" | sed '/^$/d' | wc -l | tr -d ' ') log=$unit_log"
exit $status
