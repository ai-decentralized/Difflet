#!/usr/bin/env bash
# Wan 2.1 81f tp4 `difflet compile` into empty caches: an empty --cache-dir, an empty Neuron
# compile cache, and an empty presharded weight store. The store lives under
# DIFFLET_COMPILE_CACHE (~/.cache/difflet/_shared_weights) whatever --cache-dir says, so the
# first attempt (wan_2_1_compile_empty.json, 59.5 min) reused the CLI's UMT5, DiT and VAE
# shards; LTX-2's compile (17.6 min) sharded (474 s).
#
#   setsid nohup benchmark/trn2/eval/queue_wan_compile_empty.sh > LOG 2>&1 &   (venv active)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE=trn2-eval-vc2
PY="${PYTHON_BIN:-python}"
RES="benchmark/$DIFFLET_BENCH_DEVICE"
LOG="$RES/logs"
SCRATCH="${SCRATCH:-$HOME/work/scratch/followup}"
WAIT_FOR="${WAIT_FOR:-$HOME/work/logs/queue_ltx2_extra5.log}"
OUT="$RES/wan_2_1_compile_empty2.json"
mkdir -p "$LOG" "$SCRATCH"
ts() { date -u +%FT%TZ; }
if [[ -n "$WAIT_FOR" ]]; then
  echo "[wancompile] $(ts) waiting for the final ALL_DONE line in $WAIT_FOR"
  until grep -q -E "Z ALL_DONE$" "$WAIT_FOR" 2>/dev/null; do sleep 60; done
fi
[[ -f "$OUT" ]] && { echo "[wancompile] $(ts) $OUT exists"; exit 0; }
CC="$SCRATCH/compile_cache"; NCC="$SCRATCH/compile_neuron_cc"
rm -rf "${CC:?}" "${NCC:?}"; mkdir -p "$CC" "$NCC"
CMD=$("$PY" benchmark/trn2/eval/bench_cmd.py compile wan_2_1 tp4 --cache-dir "$CC")
echo "[wancompile] $(ts) start: $CMD"
t0=$(date +%s.%N)
if env NEURON_COMPILE_CACHE_URL="$NCC" DIFFLET_SHARED_WEIGHTS_DIR="$CC/_shared_weights" \
    bash -c "$CMD" > "$LOG/wan_2_1_compile_empty2.log" 2>&1; then
  wall=$(echo "$(date +%s.%N) - $t0" | bc)
  "$PY" - "$OUT" "$wall" "$LOG/wan_2_1_compile_empty2.log" <<'PYEOF'
import json, re, sys
out, wall, log = sys.argv[1], float(sys.argv[2]), sys.argv[3]
text = open(log).read()
json.dump({"what": "difflet compile wan_2_1 tp4 81f (UMT5 + DiT + chunked VAE) into empty difflet "
                   "and Neuron compile caches and an empty presharded weight store",
           "wall_s": round(wall, 3), "wall_min": round(wall / 60, 1),
           "sharding_s": [float(x) for x in re.findall(r"Done Sharding weights in ([0-9.]+)", text)],
           "reused_preshard_lines": len(re.findall(r"Reusing pre-sharded", text))},
          open(out, "w"), indent=2)
print(open(out).read())
PYEOF
  echo "[wancompile] $(ts) done in ${wall}s"
else
  echo "[wancompile] $(ts) FAILED"; tail -20 "$LOG/wan_2_1_compile_empty2.log"
fi
rm -rf "${CC:?}" "${NCC:?}"
echo "[wancompile] $(ts) Z ALL_DONE"
