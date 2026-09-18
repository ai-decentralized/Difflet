#!/usr/bin/env bash
# Async-vs-sync serving benchmark on trn2 (2026-09-18). Per VIDEO model (the
# async Videos job API exists only for video models): start `difflet serve`
# exactly as benchmark/trn2/serve_bench.sh does (tp4, campaign shape), wait for
# /ready (startup recorded), then against the SAME server session run
#   1. the sync closed loop  (/v1/videos/sync)  -> serving/<slug>_tp4_sync.json
#   2. the async closed loop (/v1/videos job API) + one burst
#                                               -> serving/<slug>_tp4_async.json
# so the two are same-host, same-process; the committed serving/<slug>_tp4.json
# (2026-09-13/14 host) stays the historical baseline. Logs under
# benchmark/trn2/logs/serving/. Serving compiles its own artifact generation on
# first start (34 / ~90 / 103 min for LTX-2 / HunyuanVideo / Wan) -- keep the
# 4 h READY_TIMEOUT and --worker-restart-timeout 3600.
#
#   benchmark/trn2/serve_bench_async.sh <done-marker> [model ...]   (run detached)
# Env: SERVE_LEVELS (1,2,4), SERVE_REQUESTS (8), SERVE_REQUESTS_HV (6), PORT (8091),
#      SERVE_BURST (8; HunyuanVideo 6), SERVE_POLL (0.25), DIFFLET_VENV
set -u
MARKER="${1:?usage: serve_bench_async.sh <done-marker> [model ...]}"
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export DIFFLET_VENV="${DIFFLET_VENV:-$ROOT/.venv}"
export PATH="$DIFFLET_VENV/bin:$PATH"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PYTHON_BIN:-python}"
PORT="${PORT:-8091}"
LEVELS="${SERVE_LEVELS:-1,2,4}"
MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=(ltx_2 hunyuan_video wan_2_1)
OUT=benchmark/trn2/serving
LOG=benchmark/trn2/logs/serving
mkdir -p "$OUT" "$LOG"
ts() { date -u +%FT%TZ; }
fail=0

stop_server() {
  pkill -f "difflet.cli.main serve" 2>/dev/null || true
  pkill -f "difflet.cli.stage" 2>/dev/null || true
  sleep 5
}

# incident 2 of the sync pass: a 0-byte compile-cache lock left by a killed
# neuronx-cc blocks the next compile for hours ("Another process must be compiling")
stale=$(find /var/tmp/neuron-compile-cache -name "*.lock" -size 0 2>/dev/null | head -5)
[[ -n "$stale" ]] && echo "[serve] $(ts) WARNING stale 0-byte compile locks: $stale"

for m in "${MODELS[@]}"; do
  if [[ -f "$OUT/${m}_tp4_sync.json" && -f "$OUT/${m}_tp4_async.json" ]]; then
    echo "[serve] $m already measured (sync + async), skipping"; continue
  fi
  read -r mid h w f steps kind < <("$PY" -c "from benchmark.models import MATRIX; c=MATRIX['$m']; print(c.model_id, c.height, c.width, c.num_frames, c.steps, c.output_kind)")
  if [[ "$kind" != "video" ]]; then echo "[serve] $m is not a video model: no async endpoint, skipping"; continue; fi
  shape_flags=(--height "$h" --width "$w"); [[ "$f" != "None" ]] && shape_flags+=(--num-frames "$f")
  nreq="${SERVE_REQUESTS:-8}"; burst="${SERVE_BURST:-8}"
  [[ "$m" == hunyuan_video ]] && { nreq="${SERVE_REQUESTS_HV:-6}"; burst="${SERVE_BURST_HV:-6}"; }
  echo "[serve] $(ts) === $m ($mid ${h}x${w}x${f}, ${steps} steps) levels=$LEVELS requests/level=$nreq burst=$burst ==="
  stop_server
  t0=$(date +%s.%N)
  "$PY" -m difflet.cli.main serve --model-id "$mid" --tp-degree 4 "${shape_flags[@]}" \
    --port "$PORT" --request-timeout 1800 --max-queued-requests 8 --worker-restart-timeout 3600 >"$LOG/${m}_async_serve.log" 2>&1 &
  spid=$!
  t=0
  until curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 "$spid" 2>/dev/null; then echo "[serve] $m server exited before ready"; tail -20 "$LOG/${m}_async_serve.log"; break; fi
    if [[ $t -ge ${READY_TIMEOUT:-14400} ]]; then echo "[serve] $m READY_TIMEOUT"; break; fi
  done
  ready=$(echo "$(date +%s.%N) - $t0" | bc)
  if ! curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; then echo "[serve] $(ts) $m FAILED (not ready)"; stop_server; fail=1; continue; fi
  echo "[serve] $(ts) $m ready after ${ready}s"
  phase="$LOG/${m}_async_phase.txt"; echo startup > "$phase"
  "$PY" scripts/sample_serving_resources.py --pid "$spid" --output "$LOG/${m}_async_resources.jsonl" \
    --phase-file "$phase" --interval 5 >"$LOG/${m}_async_sampler.log" 2>&1 &
  sampler=$!
  if [[ ! -f "$OUT/${m}_tp4_sync.json" ]]; then
    echo "[serve] $(ts) $m sync pass (/v1/videos/sync)"
    "$PY" -m benchmark.serve_bench --model "$m" --port "$PORT" --levels "$LEVELS" --requests "$nreq" \
      --warmup 2 --api-mode sync --out "$OUT/${m}_tp4_sync.json" --phase-file "$phase" \
      --ready-seconds "$ready" 2>&1 | tee "$LOG/${m}_sync_bench.log" | grep --line-buffered "^\[serve_bench\]"
    [[ -f "$OUT/${m}_tp4_sync.json" ]] || { echo "[serve] $(ts) $m SYNC_FAILED"; fail=1; }
  fi
  echo "[serve] $(ts) $m async pass (/v1/videos job API, poll ${SERVE_POLL:-0.25}s, burst $burst)"
  "$PY" -m benchmark.serve_bench --model "$m" --port "$PORT" --levels "$LEVELS" --requests "$nreq" \
    --warmup 2 --api-mode async --poll-interval "${SERVE_POLL:-0.25}" --burst "$burst" \
    --out "$OUT/${m}_tp4_async.json" --phase-file "$phase" --sampler-jsonl "$LOG/${m}_async_resources.jsonl" \
    --ready-seconds "$ready" 2>&1 | tee "$LOG/${m}_async_bench.log" | grep --line-buffered "^\[serve_bench\]"
  [[ -f "$OUT/${m}_tp4_async.json" ]] || { echo "[serve] $(ts) $m ASYNC_FAILED"; fail=1; }
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  stop_server
  echo "[serve] $(ts) $m MODEL_DONE"
done
echo "[serve] $(ts) ALL_DONE fail=$fail" | tee "$MARKER"
exit "$fail"
