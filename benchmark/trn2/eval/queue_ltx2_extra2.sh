#!/usr/bin/env bash
# LTX-2's PyTorch/XLA attribution row, after queue_ltx2_extra.sh: the SDPA program with
# the weights sharded at load (the paper's first two attribution rows both shard at load
# so the attention kernel is separated from the loader). Hardlink clone of the cache
# without presharded shards; run twice, keep the second.
#
#   setsid nohup benchmark/trn2/eval/queue_ltx2_extra2.sh > LOG 2>&1 &   (venv active)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE=trn2-eval-vc2
PY="${PYTHON_BIN:-python}"
RES="benchmark/$DIFFLET_BENCH_DEVICE"
LOG="$RES/logs"
CACHE="$HOME/.cache/difflet"
SCRATCH="${SCRATCH:-$HOME/work/scratch/ltx2}"
WAIT_FOR="${WAIT_FOR:-$HOME/work/logs/queue_ltx2_extra.log}"
mkdir -p "$LOG" "$SCRATCH"
ts() { date -u +%FT%TZ; }
step() {  # step <name> <log> <cmd...>
  local name="$1" log="$2"; shift 2
  echo "[ltx2x2] $(ts) $name start -> $log"
  local t0=$SECONDS
  if "$@" >"$log" 2>&1; then echo "[ltx2x2] $(ts) $name done in $((SECONDS - t0))s"; return 0; fi
  echo "[ltx2x2] $(ts) $name FAILED after $((SECONDS - t0))s; tail:"; tail -n 20 "$log" | sed 's/^/    /'
  return 1
}
has() { [[ -f "$RES/$1.json" ]] && "$PY" -c "import json,sys; d=json.load(open('$RES/$1.json')); sys.exit(0 if d.get('$2') not in (None, {}, []) else 1)"; }

if [[ -n "$WAIT_FOR" ]]; then
  echo "[ltx2x2] $(ts) waiting for the final ALL_DONE line in $WAIT_FOR"
  until grep -q -E "Z ALL_DONE$" "$WAIT_FOR" 2>/dev/null; do sleep 60; done
fi

# SDPA program, weights sharded at load
NOPRE="$SCRATCH/cache_nopreshard_sdpa"
NOPRE_OUT="$RES/ltx_2_tp4sdpa_nopreshard.json"
if [[ ! -f "$NOPRE_OUT" ]]; then
  rm -rf "${NOPRE:?}"
  cp -al "$CACHE" "$NOPRE"
  rm -rf "${NOPRE:?}/_shared_weights" "${NOPRE:?}/serving" "${NOPRE:?}/work"
  find "$NOPRE" -name 'tp*_sharded_checkpoint.safetensors' -delete
  GEN=$("$PY" benchmark/trn2/eval/bench_cmd.py generate ltx_2 tp4sdpa --cache-dir "$NOPRE" --output "$SCRATCH/nopreshard_sdpa.mp4")
  walls=()
  for i in 1 2; do
    t0=$(date +%s.%N)
    if step "ltx_2_tp4sdpa_nopreshard/run$i" "$LOG/ltx_2_tp4sdpa_nopreshard_run$i.log" bash -c "$GEN"; then
      walls+=("$(echo "$(date +%s.%N) - $t0" | bc)")
    else
      walls=(); break
    fi
  done
  if [[ ${#walls[@]} -eq 2 ]]; then
    "$PY" - "$NOPRE_OUT" "${walls[@]}" "$LOG/ltx_2_tp4sdpa_nopreshard_run2.log" <<'EOF'
import json, re, sys
out, w1, w2, log = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
text = open(log).read()
json.dump({"what": "LTX-2 480x704x49 tp4 standalone, PyTorch SDPA attention, weights sharded at load (no presharded "
                   "shards); run 2 = warm page cache",
           "walls_s": [round(w1, 3), round(w2, 3)], "warm_s": round(w2, 3),
           "sharding_on_load_lines": len(re.findall(r"Sharding weights on load", text))},
          open(out, "w"), indent=2)
print(open(out).read())
EOF
  fi
  rm -rf "${NOPRE:?}"
fi
echo "[ltx2x2] $(ts) ALL_DONE"
