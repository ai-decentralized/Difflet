#!/usr/bin/env bash
# LTX-2's two shard-at-load attribution rows (PyTorch/XLA = SDPA, "+ NKI attention" = tp4)
# measured as a pair under one protocol. Of five shard-at-load runs so far, four spent
# 120-141 s in "device init" and one, the SDPA warm row (110.4 s), 7.6 s; the NKI warm row
# (262.1 s, queue_ltx2_extra4.sh) took 140.8 s there. The attention kernel is not the
# difference (run 2 of the run-twice pair: NKI 120.8 s, SDPA 120.6 s), so this alternates
# the two configurations, twice each, with the extra4 protocol (transformer read first,
# host text encoder last, run on an idle host) and records vmstat during each run.
#
#   setsid nohup benchmark/trn2/eval/queue_ltx2_extra5.sh > LOG 2>&1 &   (venv active)
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
WAIT_FOR="${WAIT_FOR:-$HOME/work/logs/queue_wan_serve_hostvae.log}"
mkdir -p "$LOG" "$SCRATCH"
ts() { date -u +%FT%TZ; }

if [[ -n "$WAIT_FOR" ]]; then
  echo "[ltx2x5] $(ts) waiting for the final ALL_DONE line in $WAIT_FOR"
  until grep -q -E "Z ALL_DONE$" "$WAIT_FOR" 2>/dev/null; do sleep 60; done
fi

SNAP=$("$PY" -c "from benchmark.models import MATRIX; from difflet.pipeline.path_resolver import resolve_model_path; c=MATRIX['ltx_2']; print(resolve_model_path(c.model_id, revision=c.revision, local_files_only=True))")
NOPRE="$SCRATCH/cache_nopreshard_pair"
OUT="$RES/ltx_2_nopreshard_pair.json"
RUNS="$SCRATCH/nopreshard_pair_runs.jsonl"
[[ -f "$OUT" ]] && { echo "[ltx2x5] $(ts) $OUT exists"; exit 0; }
: > "$RUNS"
i=0
for cfg in tp4 tp4sdpa tp4 tp4sdpa; do
  i=$((i + 1)); name="ltx_2_${cfg}_nopreshard_pair$i"
  rm -rf "${NOPRE:?}"
  cp -al "$CACHE" "$NOPRE"
  rm -rf "${NOPRE:?}/_shared_weights" "${NOPRE:?}/serving" "${NOPRE:?}/work"
  find "$NOPRE" -name 'tp*_sharded_checkpoint.safetensors' -delete
  GEN=$("$PY" benchmark/trn2/eval/bench_cmd.py generate ltx_2 "$cfg" --cache-dir "$NOPRE" --output "$SCRATCH/nopreshard_pair_$cfg.mp4")
  t0=$(date +%s.%N)
  find -L "$SNAP" -type f ! -path "*/text_encoder/*" -exec cat {} + > /dev/null
  find -L "$SNAP/text_encoder" -type f -exec cat {} + > /dev/null
  prewarm=$(echo "$(date +%s.%N) - $t0" | bc)
  echo "[ltx2x5] $(ts) $name: page cache warmed in ${prewarm}s; $(free -g | awk '/Mem:/{print "free " $4 " GB, cache " $6 " GB"}')"
  vmstat -t 5 > "$LOG/$name.vmstat" 2>&1 &
  VM=$!
  t0=$(date +%s.%N)
  if bash -c "$GEN" > "$LOG/$name.log" 2>&1; then
    wall=$(echo "$(date +%s.%N) - $t0" | bc)
    kill "$VM" 2>/dev/null
    "$PY" - "$RUNS" "$cfg" "$i" "$wall" "$prewarm" "$LOG/$name.log" <<'PYEOF'
import json, re, sys
runs, cfg, i, wall, prewarm, log = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
text = open(log).read()
init = re.search(r"device init ([0-9.]+)s, total load_weights ([0-9.]+)s", text)
row = {"cfg": cfg, "run": i, "warm_s": round(wall, 3), "prewarm_s": round(prewarm, 3),
       "device_init_s": float(init.group(1)) if init else None,
       "load_weights_s": float(init.group(2)) if init else None,
       "host_text_encoder_load": re.findall(r"Loading checkpoint shards: 100%\|[^\]]*\[(\d+:\d+)<", text),
       "sharding_on_load_lines": len(re.findall(r"Sharding weights on load", text))}
open(runs, "a").write(json.dumps(row) + "\n")
print(row)
PYEOF
    echo "[ltx2x5] $(ts) $name done in ${wall}s"
  else
    kill "$VM" 2>/dev/null
    echo "[ltx2x5] $(ts) $name FAILED; tail:"; tail -n 20 "$LOG/$name.log" | sed 's/^/    /'
  fi
  rm -rf "${NOPRE:?}"
done
"$PY" - "$RUNS" "$OUT" <<'PYEOF'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
summary = {}
for cfg in ("tp4sdpa", "tp4"):
    walls = [r["warm_s"] for r in rows if r["cfg"] == cfg]
    if walls:
        summary[cfg] = {"warm_s": walls, "median_warm_s": statistics.median(walls)}
json.dump({"what": "LTX-2 480x704x49 standalone, weights sharded at load, warm (files read into the "
                   "page cache before each run, transformer first); tp4sdpa = PyTorch/XLA row, "
                   "tp4 = + NKI attention row; alternated, twice each",
           "runs": rows, "summary": summary}, open(sys.argv[2], "w"), indent=2)
print(open(sys.argv[2]).read())
PYEOF
echo "[ltx2x5] $(ts) Z ALL_DONE"
