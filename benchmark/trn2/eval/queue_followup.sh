#!/usr/bin/env bash
# Paper eval follow-up after queue_wan_vc2.sh: the rest of the Wan 2.1 rows on the
# LNC2 attention program (26c0ffd), then the Qwen-Image true-CFG check (P11).
# Runs serially, each step skipped when its result exists (resumable).
#
#   0. Qwen-Image TP4 with true CFG: compile, cold/warm, DiT step (first, so the
#      true-CFG commit can be pushed once it reproduces the 835 ms step).
#   1. Wan step caching (denoising time): cadence 2, online-delta 0.6, 30 steps.
#   2. Wan "+ NKI attention" attribution row: warm standalone with the weights
#      sharded at load (a hardlink clone of the cache without presharded shards).
#   3. Wan resident serving: first start into empty caches, resident request
#      latency (serve_bench, 1 warm-up + 3), warm and cold restarts.
#   4. Wan standalone compile into empty caches (UMT5 + DiT + chunked VAE).
#
#   setsid nohup benchmark/trn2/eval/queue_followup.sh > LOG 2>&1 &   (venv active)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PYTHON_BIN:-python}"
CACHE="$HOME/.cache/difflet"
SCRATCH="${SCRATCH:-$HOME/work/scratch/followup}"
WAIT_FOR="${WAIT_FOR:-$HOME/work/logs/queue_ltx2.log}"
mkdir -p "$SCRATCH"
ts() { date -u +%FT%TZ; }
step() {  # step <name> <log> <cmd...>
  local name="$1" log="$2"; shift 2
  echo "[followup] $(ts) $name start -> $log"
  local t0=$SECONDS
  if "$@" >"$log" 2>&1; then echo "[followup] $(ts) $name done in $((SECONDS - t0))s"; return 0; fi
  echo "[followup] $(ts) $name FAILED after $((SECONDS - t0))s; tail:"; tail -n 20 "$log" | sed 's/^/    /'
  return 1
}
drop_caches() { sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"; }
# has <slug> <field>: the result JSON under the current $RES carries <field>
has() { [[ -f "$RES/$1.json" ]] && "$PY" -c "import json,sys; d=json.load(open('$RES/$1.json')); sys.exit(0 if d.get('$2') not in (None, {}, []) else 1)"; }

if [[ -n "$WAIT_FOR" ]]; then
  echo "[followup] $(ts) waiting for ALL_DONE in $WAIT_FOR"
  # the final line only ("... <UTC>Z ALL_DONE"), not a "waiting for ALL_DONE" banner
  until grep -q -E "Z ALL_DONE$" "$WAIT_FOR" 2>/dev/null; do sleep 60; done
fi

# ---------------------------------------------------------------- Qwen-Image, true CFG (P11)
export DIFFLET_BENCH_DEVICE=trn2-eval
RES="benchmark/$DIFFLET_BENCH_DEVICE"
LOG="$RES/logs"
mkdir -p "$LOG"
s=qwen_image
has "$s" compile_seconds || step "$s/compile" "$LOG/${s}_compile.log" \
  "$PY" -m benchmark.bench --model qwen_image --config tp4 --skip-download --compile-only
if has "$s" compile_seconds; then
  has "$s" e2e_warm || step "$s/cold_warm" "$LOG/${s}_cold_warm.log" \
    "$PY" -m benchmark.cold_warm_e2e --model qwen_image --config tp4
  has "$s" step_latency || step "$s/step" "$LOG/${s}_realloop.log" \
    "$PY" -m benchmark.step_realloop --model qwen_image --config tp4
fi

# ---------------------------------------------------------------- Wan, LNC2 program
export DIFFLET_BENCH_DEVICE=trn2-eval-vc2
RES="benchmark/$DIFFLET_BENCH_DEVICE"
LOG="$RES/logs"
mkdir -p "$LOG" "$RES/serving"

# 1. step caching: the tp4 artifact (runtime-only overlays), denoising time = loop_wall_s
for cfg in tp4tc2 tp4tcod tp4s30; do
  s="wan_2_1_$cfg"
  has "$s" compile_seconds || step "$s/compile" "$LOG/${s}_compile.log" \
    "$PY" -m benchmark.bench --model wan_2_1 --config "$cfg" --skip-download --compile-only || continue
  has "$s" loop_wall_s || step "$s/step" "$LOG/${s}_realloop.log" \
    "$PY" -m benchmark.step_realloop --model wan_2_1 --config "$cfg"
done

# 2. weights sharded at load: clone the cache with hardlinks, drop every presharded
#    shard and the shared weight store from the clone, run twice, keep the second
NOPRE="$SCRATCH/cache_nopreshard"
NOPRE_OUT="$RES/wan_2_1_nopreshard.json"
if [[ ! -f "$NOPRE_OUT" ]]; then
  rm -rf "$NOPRE"
  cp -al "$CACHE" "$NOPRE"
  rm -rf "$NOPRE/_shared_weights" "$NOPRE/serving" "$NOPRE/work"
  find "$NOPRE" -name 'tp*_sharded_checkpoint.safetensors' -delete
  GEN=$("$PY" benchmark/trn2/eval/bench_cmd.py generate wan_2_1 tp4 --cache-dir "$NOPRE" --output "$SCRATCH/nopreshard.mp4")
  walls=()
  for i in 1 2; do
    t0=$(date +%s.%N)
    if step "wan_2_1_nopreshard/run$i" "$LOG/wan_2_1_nopreshard_run$i.log" bash -c "$GEN"; then
      walls+=("$(echo "$(date +%s.%N) - $t0" | bc)")
    else
      walls=(); break
    fi
  done
  if [[ ${#walls[@]} -eq 2 ]]; then
    "$PY" - "$NOPRE_OUT" "${walls[@]}" "$LOG/wan_2_1_nopreshard_run2.log" <<'EOF'
import json, re, sys
out, w1, w2, log = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
text = open(log).read()
json.dump({"what": "Wan 2.1 81f tp4 standalone, weights sharded at load (no presharded shards); "
                   "run 2 = warm page cache",
           "walls_s": [round(w1, 3), round(w2, 3)], "warm_s": round(w2, 3),
           "sharding_on_load_lines": len(re.findall(r"Sharding weights on load", text))},
          open(out, "w"), indent=2)
print(open(out).read())
EOF
  fi
  rm -rf "$NOPRE"
fi

# 3. resident serving (difflet serve), into empty caches first. The Neuron decoder is
#    the chunked FP32 VAE (the CLI's --wan-vae-chunked graph) on one core; the unchunked
#    81-frame decoder exceeds the compiler's instruction limit (39.1 M vs 10 M).
SERVE_OUT="$RES/serving/wan_2_1_resident.json"
if [[ ! -f "$SERVE_OUT" ]]; then
  SC="$SCRATCH/serve_cache"; NCC="$SCRATCH/serve_neuron_cc"; PORT=8093
  read -r mid h w f < <("$PY" -c "from benchmark.models import MATRIX; c=MATRIX['wan_2_1']; print(c.model_id, c.height, c.width, c.num_frames)")
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
  # serve() runs in a command substitution, so its $SERVE_PID never reaches this shell
  stop_serve() { pkill -f "difflet.cli.main serve" 2>/dev/null; pkill -f "difflet.cli.stage" 2>/dev/null; sleep 10; }
  rm -rf "$SC" "$NCC"; mkdir -p "$SC" "$NCC"
  echo "[followup] $(ts) wan serve first start (empty caches)"
  if first=$(serve "$LOG/wan_2_1_serve_first.log"); then
    echo "[followup] $(ts) ready after ${first}s"
    sleep 90  # the device recovers from back-to-back load after 90 s idle
    step "wan_2_1/resident" "$LOG/wan_2_1_serve_bench.log" "$PY" -m benchmark.serve_bench \
      --model wan_2_1 --port "$PORT" --levels 1 --requests 3 --warmup 1 --timeout 3600 \
      --out "$RES/serving/wan_2_1_serve_bench.json"
    stop_serve
    warm=$(serve "$LOG/wan_2_1_serve_restart_warm.log"); stop_serve
    drop_caches
    cold=$(serve "$LOG/wan_2_1_serve_restart_cold.log"); stop_serve
    "$PY" - "$SERVE_OUT" "$first" "${warm:-nan}" "${cold:-nan}" "$RES/serving/wan_2_1_serve_bench.json" <<'EOF'
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

# 4. standalone compile into empty caches (difflet + Neuron compile caches)
COMPILE_OUT="$RES/wan_2_1_compile_empty.json"
if [[ ! -f "$COMPILE_OUT" ]]; then
  CC="$SCRATCH/compile_cache"; NCC="$SCRATCH/compile_neuron_cc"
  rm -rf "$CC" "$NCC"; mkdir -p "$CC" "$NCC"
  CMD=$("$PY" benchmark/trn2/eval/bench_cmd.py compile wan_2_1 tp4 --cache-dir "$CC")
  t0=$(date +%s.%N)
  if step "wan_2_1/compile_empty" "$LOG/wan_2_1_compile_empty.log" env NEURON_COMPILE_CACHE_URL="$NCC" bash -c "$CMD"; then
    wall=$(echo "$(date +%s.%N) - $t0" | bc)
    "$PY" -c "import json; json.dump({'what': 'difflet compile wan_2_1 tp4 81f (UMT5 + DiT + chunked VAE) into empty difflet and Neuron compile caches', 'wall_s': $wall, 'wall_min': round($wall / 60, 1)}, open('$COMPILE_OUT', 'w'), indent=2)"
    cat "$COMPILE_OUT"
  fi
  rm -rf "$CC" "$NCC"
fi

echo "[followup] $(ts) ALL_DONE"
