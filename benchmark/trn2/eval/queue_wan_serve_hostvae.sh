#!/usr/bin/env bash
# Wan 2.1 resident serving rows with the VAE decode on the host (--host-vae), the setting
# of the paper's original resident Wan row.
#
# The chunked FP32 decoder (the CLI's --wan-vae-chunked graph) does not fit beside the
# 81-frame generation artifact on trn2.3xlarge: UMT5 + DiT occupy ~20.5 of each logical
# core's 24 GB HBM (5.5 GB of it the DiT's DMA-ring spill, 3.7 GB scratchpad), and the
# decoder's steady-state tail bucket alone moves ~3 GB of FP32 causal state in and ~3 GB
# out (first decode: "Failed to allocate DEVICE memory" on ND 0:NC 4).
#
# First start into empty caches, resident request latency (serve_bench, 1 warm-up + 3),
# warm and cold restarts. "Empty" includes the presharded weight store: it lives under
# DIFFLET_COMPILE_CACHE (~/.cache/difflet/_shared_weights) whatever --cache-dir says, and
# the serving transformer's store key equals the CLI's, so without
# DIFFLET_SHARED_WEIGHTS_DIR the first start reuses the CLI's shards (LTX-2's serving key
# differs, so its first start sharded: 476.7 s).
#
#   setsid nohup benchmark/trn2/eval/queue_wan_serve_hostvae.sh > LOG 2>&1 &   (venv active)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE=trn2-eval-vc2
PY="${PYTHON_BIN:-python}"
RES="benchmark/$DIFFLET_BENCH_DEVICE"
LOG="$RES/logs"
SCRATCH="${SCRATCH:-$HOME/work/scratch/followup}"
mkdir -p "$LOG" "$RES/serving" "$SCRATCH"
ts() { date -u +%FT%TZ; }
step() {  # step <name> <log> <cmd...>
  local name="$1" log="$2"; shift 2
  echo "[wanserve] $(ts) $name start -> $log"
  local t0=$SECONDS
  if "$@" >"$log" 2>&1; then echo "[wanserve] $(ts) $name done in $((SECONDS - t0))s"; return 0; fi
  echo "[wanserve] $(ts) $name FAILED after $((SECONDS - t0))s; tail:"; tail -n 20 "$log" | sed 's/^/    /'
  return 1
}
drop_caches() { sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"; }

SERVE_OUT="$RES/serving/wan_2_1_resident.json"
if [[ ! -f "$SERVE_OUT" ]]; then
  SC="$SCRATCH/serve_cache"; NCC="$SCRATCH/serve_neuron_cc"; PORT=8093
  read -r mid h w f < <("$PY" -c "from benchmark.models import MATRIX; c=MATRIX['wan_2_1']; print(c.model_id, c.height, c.width, c.num_frames)")
  serve() {  # serve <log>: start the server, wait for /ready, print seconds to ready
    local log="$1" t0 t
    t0=$(date +%s.%N)
    NEURON_COMPILE_CACHE_URL="$NCC" DIFFLET_SHARED_WEIGHTS_DIR="$SC/_shared_weights" \
      "$PY" -m difflet.cli.main serve --model-id "$mid" --tp-degree 4 \
      --height "$h" --width "$w" --num-frames "$f" --cache-dir "$SC" --port "$PORT" --host-vae \
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
  # serve() runs in a command substitution, so its $SERVE_PID never reaches this shell
  stop_serve() { pkill -f "difflet.cli.main serve" 2>/dev/null; pkill -f "difflet.cli.stage" 2>/dev/null; sleep 10; }
  rm -rf "${SC:?}" "${NCC:?}"; mkdir -p "$SC" "$NCC"
  echo "[wanserve] $(ts) wan serve first start (empty caches, host VAE)"
  if first=$(serve "$LOG/wan_2_1_serve_first.log"); then
    echo "[wanserve] $(ts) ready after ${first}s"
    sleep 90  # the device recovers from back-to-back load after 90 s idle
    step "wan_2_1/resident" "$LOG/wan_2_1_serve_bench.log" "$PY" -m benchmark.serve_bench \
      --model wan_2_1 --port "$PORT" --levels 1 --requests 3 --warmup 1 --timeout 3600 \
      --out "$RES/serving/wan_2_1_serve_bench.json"
    stop_serve
    warm=$(serve "$LOG/wan_2_1_serve_restart_warm.log"); echo "[wanserve] $(ts) restart warm ${warm:-FAILED}s"; stop_serve
    drop_caches
    cold=$(serve "$LOG/wan_2_1_serve_restart_cold.log"); echo "[wanserve] $(ts) restart cold ${cold:-FAILED}s"; stop_serve
    "$PY" - "$SERVE_OUT" "$first" "${warm:-nan}" "${cold:-nan}" "$RES/serving/wan_2_1_serve_bench.json" <<'PYEOF'
import json, sys
out, first, warm, cold, bench = sys.argv[1:6]
b = json.load(open(bench))
json.dump({"decoder": "host VAE (--host-vae)",
           "first_start_s": float(first), "first_start_min": round(float(first) / 60, 1),
           "restart_warm_s": float(warm), "restart_cold_s": float(cold),
           "serve_bench": bench, "levels": b.get("levels", b)},
          open(out, "w"), indent=2, default=str)
print(open(out).read())
PYEOF
  else
    stop_serve
  fi
  rm -rf "${SC:?}" "${NCC:?}"
fi
echo "[wanserve] $(ts) Z ALL_DONE"
