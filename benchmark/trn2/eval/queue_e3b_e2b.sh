#!/usr/bin/env bash
# Paper eval queue after E5b (docs/eval-redesign-plan-20261003.md): one device,
# so every job runs serially, cheapest first, and each step is skipped when its
# result already exists (a killed queue resumes where it stopped).
#
#   1. E3b correctness: masked Ulysses vs unsplit attention, HunyuanVideo DiT
#      (2+2 blocks, 320x512x61), five text lengths, one program per mode.
#   2. E2b: Wan 2.1 at 9 / 33 frames, NKI and SDPA, 8-step loop (compile +
#      DiT step only; per-step does not depend on the step count).
#   3. E3b: Wan 2.1 at 81 frames, tp4+sp and tp2cp2 (ulysses): compile, cold and
#      warm standalone latency, DiT step.
#   4. Wan 2.1 tp4 at 81 frames on this host, the baseline both sweeps
#      normalise to (the paper's tp4 numbers came from the lost campaign).
#
# Results: benchmark/trn2-eval/<slug>_<config>.json (DIFFLET_BENCH_DEVICE=trn2-eval
# keeps them apart from the 2026-09 campaign files in benchmark/trn2/) and
# benchmark/trn2/eval/e3b/hv_ulysses_mask_parity.json.
#
#   setsid nohup benchmark/trn2/eval/queue_e3b_e2b.sh > LOG 2>&1 &   (venv active)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE=trn2-eval
PY="${PYTHON_BIN:-python}"
RES="benchmark/$DIFFLET_BENCH_DEVICE"
LOG="$RES/logs"
E3B=benchmark/trn2/eval/e3b
mkdir -p "$LOG" "$E3B"
ts() { date -u +%FT%TZ; }
has() { [[ -f "$RES/$1.json" ]] && "$PY" -c "import json,sys; d=json.load(open('$RES/$1.json')); sys.exit(0 if d.get('$2') not in (None, {}, []) else 1)"; }
step() {  # step <name> <log> <cmd...>
  local name="$1" log="$2"; shift 2
  echo "[queue] $(ts) $name start -> $log"
  local t0=$SECONDS
  if "$@" >"$log" 2>&1; then echo "[queue] $(ts) $name done in $((SECONDS - t0))s"; return 0; fi
  echo "[queue] $(ts) $name FAILED after $((SECONDS - t0))s; tail:"; tail -n 20 "$log" | sed 's/^/    /'
  return 1
}

# 1. masked Ulysses correctness
P="scripts/hunyuan_ulysses_mask_parity.py"
if [[ ! -f "$E3B/hv_ulysses_mask_parity.json" ]]; then
  for mode in reference ulysses; do
    [[ -f "$E3B/hv_parity_$mode.pt" ]] || step "parity/$mode" "$LOG/hv_parity_$mode.log" \
      "$PY" "$P" --mode "$mode" --out "$E3B/hv_parity_$mode.pt"
  done
  [[ -f "$E3B/hv_parity_reference.pt" && -f "$E3B/hv_parity_ulysses.pt" ]] && step parity/compare \
    "$LOG/hv_parity_compare.log" "$PY" "$P" --compare "$E3B/hv_parity_reference.pt" \
    "$E3B/hv_parity_ulysses.pt" --json "$E3B/hv_ulysses_mask_parity.json"
fi

# 2. attention vs sequence length
for cfg in tp4f9 tp4f9sdpa tp4f33 tp4f33sdpa; do
  s="wan_2_1_$cfg"
  if ! has "$s" compile_seconds; then
    step "$s/compile" "$LOG/${s}_compile.log" \
      "$PY" -m benchmark.bench --model wan_2_1 --config "$cfg" --skip-download --compile-only || continue
  fi
  has "$s" step_latency || step "$s/step" "$LOG/${s}_realloop.log" \
    "$PY" -m benchmark.step_realloop --model wan_2_1 --config "$cfg"
done

# 3 + 4. parallel configurations at 81 frames, then the tp4 baseline
for cfg in tp4sp tp2cp2 tp4; do
  [[ "$cfg" == tp4 ]] && s=wan_2_1 || s="wan_2_1_$cfg"
  if ! has "$s" compile_seconds; then
    step "$s/compile" "$LOG/${s}_compile.log" \
      "$PY" -m benchmark.bench --model wan_2_1 --config "$cfg" --skip-download --compile-only || continue
  fi
  has "$s" e2e_warm || step "$s/cold_warm" "$LOG/${s}_cold_warm.log" \
    "$PY" -m benchmark.cold_warm_e2e --model wan_2_1 --config "$cfg"
  has "$s" step_latency || step "$s/step" "$LOG/${s}_realloop.log" \
    "$PY" -m benchmark.step_realloop --model wan_2_1 --config "$cfg"
done
echo "[queue] $(ts) ALL_DONE"
