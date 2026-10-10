#!/usr/bin/env bash
# E5b (paper eval, docs/eval-redesign-plan-20261003.md 7.6): open-loop Poisson
# load through `difflet serve` over HTTP. LAYOUT=tp4 starts one TP4 server;
# LAYOUT=dp2tp2 starts two TP2 servers pinned to separate core pairs and routes
# each arrival to the one with fewer requests in flight. Both layouts get the
# same arrival sequence (seeds ARRIVAL_SEED, +1, +2 per level).
#
#   benchmark/trn2/eval/e5b_load.sh <model> [tp4|dp2tp2]     (venv active; run detached)
# Env: RHOS (0.5,0.8,0.95), ARRIVALS (20), SLO (s; default 2.1 x REF_LATENCY,
#      the FLUX ratio 30 / 14.2), REF_LATENCY (s; default: TP4 warm-up median),
#      PORT (8091), READY_TIMEOUT (14400), COOLDOWN (90)
# Result: benchmark/trn2/eval/e5b/<model>_<layout>_<UTC stamp>.json; logs beside it.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PYTHON_BIN:-python}"
MODEL="${1:?model slug}"
LAYOUT="${2:-tp4}"
PORT="${PORT:-8091}"
STAMP="$(date -u +%Y%m%dT%H%M)"
OUT=benchmark/trn2/eval/e5b
LOG="$OUT/logs/${MODEL}_${LAYOUT}_${STAMP}"
mkdir -p "$LOG"
ts() { date -u +%FT%TZ; }

stop_servers() {
  pkill -f "difflet.cli.main serve" 2>/dev/null || true
  pkill -f "difflet.cli.stage" 2>/dev/null || true
  sleep 5
}
trap stop_servers EXIT

read -r mid h w f < <("$PY" -c "from benchmark.models import MATRIX; c=MATRIX['$MODEL']; print(c.model_id, c.height, c.width, c.num_frames)")
shape=(--height "$h" --width "$w"); [[ "$f" != "None" ]] && shape+=(--num-frames "$f")

case "$LAYOUT" in
  tp4)    tps=(4);   cores=("0-3");       ports=("$PORT") ;;
  dp2tp2) tps=(2 2); cores=("0-1" "2-3"); ports=("$PORT" "$((PORT+1))") ;;
  *) echo "unknown layout $LAYOUT" >&2; exit 2 ;;
esac

stop_servers
pids=()
t0=$(date +%s.%N)
for i in "${!tps[@]}"; do
  NEURON_RT_VISIBLE_CORES="${cores[$i]}" "$PY" -m difflet.cli.main serve --model-id "$mid" \
    --tp-degree "${tps[$i]}" "${shape[@]}" --port "${ports[$i]}" --request-timeout 3600 \
    --max-queued-requests 32 --worker-restart-timeout 3600 >"$LOG/serve_$i.log" 2>&1 &
  pids+=($!)
  # the second server compiles nothing new (same TP2 program) but loading two
  # at once doubles host memory pressure: wait for each before the next
  t=0
  until curl -sf "http://127.0.0.1:${ports[$i]}/ready" >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 "${pids[$i]}" 2>/dev/null; then echo "[e5b] server $i exited before ready"; tail -30 "$LOG/serve_$i.log"; exit 1; fi
    if [[ $t -ge ${READY_TIMEOUT:-14400} ]]; then echo "[e5b] server $i READY_TIMEOUT"; exit 1; fi
  done
  echo "[e5b] $(ts) server $i (tp${tps[$i]}, cores ${cores[$i]}, port ${ports[$i]}) ready"
done
ready=$(echo "$(date +%s.%N) - $t0" | bc)

extra=()
[[ -n "${REF_LATENCY:-}" ]] && extra+=(--ref-latency "$REF_LATENCY")
if [[ -n "${SLO:-}" ]]; then extra+=(--slo "$SLO")
elif [[ -n "${REF_LATENCY:-}" ]]; then extra+=(--slo "$(echo "2.1 * $REF_LATENCY" | bc)")
fi
"$PY" -m benchmark.serve_bench --model "$MODEL" --config tp4 --arrival poisson \
  --ports "$(IFS=,; echo "${ports[*]}")" --rhos "${RHOS:-0.5,0.8,0.95}" \
  --arrivals "${ARRIVALS:-20}" --arrival-seed "${ARRIVAL_SEED:-42}" --cooldown "${COOLDOWN:-90}" \
  --warmup 2 --ready-seconds "$ready" "${extra[@]}" \
  --out "$OUT/${MODEL}_${LAYOUT}_${STAMP}.json" 2>&1 | tee "$LOG/bench.log"
echo "[e5b] $(ts) $MODEL $LAYOUT DONE"
