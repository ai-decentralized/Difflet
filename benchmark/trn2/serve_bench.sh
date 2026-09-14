#!/usr/bin/env bash
# Serving-layer benchmark on trn2: for each model, start `difflet serve` (tp4,
# the campaign shape, existing artifact), wait for /ready (recorded as the
# startup time), sample Neuron/host resources, run the closed-loop load test
# (benchmark.serve_bench) and stop the server. Results:
# benchmark/trn2/serving/<slug>_tp4.json; logs under benchmark/trn2/logs/serving/.
#
#   benchmark/trn2/serve_bench.sh [model ...]     (venv activated; run detached)
# Env: SERVE_LEVELS (1,2,4), SERVE_REQUESTS (8), SERVE_REQUESTS_HV (6), PORT (8091)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PYTHON_BIN:-python}"
PORT="${PORT:-8091}"
LEVELS="${SERVE_LEVELS:-1,2,4}"
MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=(flux_1_dev qwen_image ltx_2 wan_2_1 hunyuan_video)
OUT=benchmark/trn2/serving
LOG=benchmark/trn2/logs/serving
mkdir -p "$OUT" "$LOG"
ts() { date -u +%FT%TZ; }

stop_server() {
  pkill -f "difflet.cli.main serve" 2>/dev/null || true
  pkill -f "difflet.cli.stage" 2>/dev/null || true
  sleep 5
}

for m in "${MODELS[@]}"; do
  if [[ -f "$OUT/${m}_tp4.json" ]]; then echo "[serve] $m already measured, skipping"; continue; fi
  read -r mid h w f steps < <("$PY" -c "from benchmark.models import MATRIX; c=MATRIX['$m']; print(c.model_id, c.height, c.width, c.num_frames, c.steps)")
  shape_flags=(--height "$h" --width "$w"); [[ "$f" != "None" ]] && shape_flags+=(--num-frames "$f")
  nreq="${SERVE_REQUESTS:-8}"; [[ "$m" == hunyuan_video ]] && nreq="${SERVE_REQUESTS_HV:-6}"
  echo "[serve] === $m ($mid ${h}x${w}x${f}, ${steps} steps) levels=$LEVELS requests/level=$nreq ==="
  stop_server
  t0=$(date +%s.%N)
  "$PY" -m difflet.cli.main serve --model-id "$mid" --tp-degree 4 "${shape_flags[@]}" \
    --port "$PORT" --request-timeout 1800 --max-queued-requests 8 --worker-restart-timeout 3600 >"$LOG/${m}_serve.log" 2>&1 &
  spid=$!
  t=0
  until curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 "$spid" 2>/dev/null; then echo "[serve] $m server exited before ready"; tail -20 "$LOG/${m}_serve.log"; break; fi
    if [[ $t -ge ${READY_TIMEOUT:-14400} ]]; then echo "[serve] $m READY_TIMEOUT"; break; fi
  done
  ready=$(echo "$(date +%s.%N) - $t0" | bc)
  if ! curl -sf "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; then echo "[serve] $m FAILED (not ready)"; stop_server; continue; fi
  echo "[serve] $(ts) $m ready after ${ready}s"
  phase="$LOG/${m}_phase.txt"; echo startup > "$phase"
  "$PY" scripts/sample_serving_resources.py --pid "$spid" --output "$LOG/${m}_resources.jsonl" \
    --phase-file "$phase" --interval 5 >"$LOG/${m}_sampler.log" 2>&1 &
  sampler=$!
  "$PY" -m benchmark.serve_bench --model "$m" --port "$PORT" --levels "$LEVELS" --requests "$nreq" \
    --warmup 2 --out "$OUT/${m}_tp4.json" --phase-file "$phase" --sampler-jsonl "$LOG/${m}_resources.jsonl" \
    --ready-seconds "$ready" 2>&1 | tee "$LOG/${m}_bench.log" | grep --line-buffered "^\[serve_bench\]"
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  stop_server
  echo "[serve] $(ts) $m DONE"
done
echo "[serve] ALL_DONE"
