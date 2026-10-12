#!/usr/bin/env bash
# LTX-2 rows re-measured on the LNC2 attention program (9e48164: DiT traced with
# NEURON_RT_VIRTUAL_CORE_SIZE=2 + strided DMA on the CLI and, newly, the serving
# path). Runs after queue_followup.sh, serially, each step skipped when done.
#
#   1. CLI: compile (LTX-2 was never built on this host, so this is a build into
#      empty caches), cold and warm standalone, DiT step.
#   2. Step caching (denoising time): cadence 2, online-delta 0.6, 25 of 40 steps.
#      Calibrated adaptive keeps the paper's values (its 40-step calibration was
#      lost; no recalibration).
#   3. Resident serving: first start into empty caches, resident request latency
#      (serve_bench, 1 warm-up + 3), warm and cold restarts.
#
#   setsid nohup benchmark/trn2/eval/queue_ltx2.sh > LOG 2>&1 &   (venv active)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE=trn2-eval-vc2
PY="${PYTHON_BIN:-python}"
RES="benchmark/$DIFFLET_BENCH_DEVICE"
LOG="$RES/logs"
SCRATCH="${SCRATCH:-$HOME/work/scratch/ltx2}"
WAIT_FOR="${WAIT_FOR:-$HOME/work/logs/queue_followup.log}"
mkdir -p "$LOG" "$RES/serving" "$SCRATCH"
ts() { date -u +%FT%TZ; }
step() {  # step <name> <log> <cmd...>
  local name="$1" log="$2"; shift 2
  echo "[ltx2] $(ts) $name start -> $log"
  local t0=$SECONDS
  if "$@" >"$log" 2>&1; then echo "[ltx2] $(ts) $name done in $((SECONDS - t0))s"; return 0; fi
  echo "[ltx2] $(ts) $name FAILED after $((SECONDS - t0))s; tail:"; tail -n 20 "$log" | sed 's/^/    /'
  return 1
}
has() { [[ -f "$RES/$1.json" ]] && "$PY" -c "import json,sys; d=json.load(open('$RES/$1.json')); sys.exit(0 if d.get('$2') not in (None, {}, []) else 1)"; }
drop_caches() { sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"; }

if [[ -n "$WAIT_FOR" ]]; then
  echo "[ltx2] $(ts) waiting for ALL_DONE in $WAIT_FOR"
  # the final line only ("... <UTC>Z ALL_DONE"), not the "waiting for ALL_DONE" banner
  until grep -q -E "Z ALL_DONE$" "$WAIT_FOR" 2>/dev/null; do sleep 60; done
fi

# 1. CLI rows
s=ltx_2
has "$s" compile_seconds || step "$s/compile" "$LOG/${s}_compile.log" \
  "$PY" -m benchmark.bench --model ltx_2 --config tp4 --skip-download --compile-only
if has "$s" compile_seconds; then
  has "$s" e2e_warm || step "$s/cold_warm" "$LOG/${s}_cold_warm.log" \
    "$PY" -m benchmark.cold_warm_e2e --model ltx_2 --config tp4
  has "$s" step_latency || step "$s/step" "$LOG/${s}_realloop.log" \
    "$PY" -m benchmark.step_realloop --model ltx_2 --config tp4
fi

# 2. step caching on the tp4 artifact (runtime-only overlays)
for cfg in tp4tc2 tp4tcod tp4s25; do
  s="ltx_2_$cfg"
  has "$s" compile_seconds || step "$s/compile" "$LOG/${s}_compile.log" \
    "$PY" -m benchmark.bench --model ltx_2 --config "$cfg" --skip-download --compile-only || continue
  has "$s" loop_wall_s || step "$s/step" "$LOG/${s}_realloop.log" \
    "$PY" -m benchmark.step_realloop --model ltx_2 --config "$cfg"
done

# 3. resident serving, into empty caches first
SERVE_OUT="$RES/serving/ltx_2_resident.json"
if [[ ! -f "$SERVE_OUT" ]]; then
  SC="$SCRATCH/serve_cache"; NCC="$SCRATCH/serve_neuron_cc"; PORT=8094
  read -r mid h w f < <("$PY" -c "from benchmark.models import MATRIX; c=MATRIX['ltx_2']; print(c.model_id, c.height, c.width, c.num_frames)")
  serve() {  # serve <log>: start the server, wait for /ready, print seconds to ready
    local log="$1" t0 t
    t0=$(date +%s.%N)
    NEURON_COMPILE_CACHE_URL="$NCC" "$PY" -m difflet.cli.main serve --model-id "$mid" --tp-degree 4 \
      --height "$h" --width "$w" --num-frames "$f" --cache-dir "$SC" --port "$PORT" \
      --request-timeout 3600 --max-queued-requests 8 --worker-restart-timeout 7200 >"$log" 2>&1 &
    SERVE_PID=$!
    t=0
    until curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; do
      sleep 5; t=$((t+5))
      if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "server exited" >&2; return 1; fi
      [[ $t -ge 14400 ]] && { echo "ready timeout" >&2; return 1; }
    done
    echo "$(date +%s.%N) - $t0" | bc
  }
  stop_serve() { pkill -f "difflet.cli.main serve" 2>/dev/null; pkill -f "difflet.cli.stage" 2>/dev/null; sleep 10; }
  rm -rf "$SC" "$NCC"; mkdir -p "$SC" "$NCC"
  echo "[ltx2] $(ts) ltx_2 serve first start (empty caches)"
  if first=$(serve "$LOG/ltx_2_serve_first.log"); then
    echo "[ltx2] $(ts) ready after ${first}s"
    sleep 90  # the device recovers from back-to-back load after 90 s idle
    step "ltx_2/resident" "$LOG/ltx_2_serve_bench.log" "$PY" -m benchmark.serve_bench \
      --model ltx_2 --port "$PORT" --levels 1 --requests 3 --warmup 1 --timeout 3600 \
      --out "$RES/serving/ltx_2_serve_bench.json"
    stop_serve
    warm=$(serve "$LOG/ltx_2_serve_restart_warm.log"); stop_serve
    drop_caches
    cold=$(serve "$LOG/ltx_2_serve_restart_cold.log"); stop_serve
    "$PY" - "$SERVE_OUT" "$first" "${warm:-nan}" "${cold:-nan}" "$RES/serving/ltx_2_serve_bench.json" <<'EOF'
import json, sys
out, first, warm, cold, bench = sys.argv[1:6]
b = json.load(open(bench))
json.dump({"first_start_s": float(first), "first_start_min": round(float(first) / 60, 1),
           "restart_warm_s": float(warm), "restart_cold_s": float(cold),
           "serve_bench": bench, "levels": b.get("levels", b)},
          open(out, "w"), indent=2, default=str)
print(open(out).read())
EOF
  else
    stop_serve
  fi
  rm -rf "$SC" "$NCC"
fi
echo "[ltx2] $(ts) ALL_DONE"
