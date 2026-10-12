#!/usr/bin/env bash
# LTX-2's shard-at-load "+ NKI attention" row again, warm by the paper's definition. The
# first warm attempt (queue_ltx2_extra3.sh, 487.8 s) read the whole snapshot alphabetically,
# so the 38 GB transformer read after the text encoder evicted it again (Gemma load 4 min
# 23 s); the SDPA run right after found it cached (3 s, 110.4 s). Here the transformer is
# read first and the text encoder last, and the run starts on an idle host.
#
# Run twice, keep the second (queue_ltx2_extra*.sh) does not give that here: sharding
# the 19B transformer on the host needs so much memory that the 124 GB host evicts the
# host text encoder (Gemma, 11 shards) during each run, so the next run re-reads it
# from EBS (6 min 10 s instead of 3 s). So each run here first reads the model's files
# into the page cache, then runs once.
#
#   setsid nohup benchmark/trn2/eval/queue_ltx2_extra4.sh > LOG 2>&1 &   (venv active)
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
WAIT_FOR="${WAIT_FOR:-$HOME/work/logs/queue_followup2.log}"
mkdir -p "$LOG" "$SCRATCH"
ts() { date -u +%FT%TZ; }
step() {  # step <name> <log> <cmd...>
  local name="$1" log="$2"; shift 2
  echo "[ltx2x4] $(ts) $name start -> $log"
  local t0=$SECONDS
  if "$@" >"$log" 2>&1; then echo "[ltx2x4] $(ts) $name done in $((SECONDS - t0))s"; return 0; fi
  echo "[ltx2x4] $(ts) $name FAILED after $((SECONDS - t0))s; tail:"; tail -n 20 "$log" | sed 's/^/    /'
  return 1
}

if [[ -n "$WAIT_FOR" ]]; then
  echo "[ltx2x4] $(ts) waiting for the final ALL_DONE line in $WAIT_FOR"
  until grep -q -E "Z ALL_DONE$" "$WAIT_FOR" 2>/dev/null; do sleep 60; done
fi

SNAP=$("$PY" -c "from benchmark.models import MATRIX; from difflet.pipeline.path_resolver import resolve_model_path; c=MATRIX['ltx_2']; print(resolve_model_path(c.model_id, revision=c.revision, local_files_only=True))")
NOPRE="$SCRATCH/cache_nopreshard_warm2"
for cfg in tp4; do
  out="$RES/ltx_2_${cfg}_nopreshard_warm2.json"
  [[ -f "$out" ]] && continue
  rm -rf "${NOPRE:?}"
  cp -al "$CACHE" "$NOPRE"
  rm -rf "${NOPRE:?}/_shared_weights" "${NOPRE:?}/serving" "${NOPRE:?}/work"
  find "$NOPRE" -name 'tp*_sharded_checkpoint.safetensors' -delete
  GEN=$("$PY" benchmark/trn2/eval/bench_cmd.py generate ltx_2 "$cfg" --cache-dir "$NOPRE" --output "$SCRATCH/nopreshard_warm_$cfg.mp4")
  t0=$(date +%s.%N)
  # the transformer first, the host text encoder (read first by the run) last
  find -L "$SNAP" -type f ! -path "*/text_encoder/*" -exec cat {} + > /dev/null
  find -L "$SNAP/text_encoder" -type f -exec cat {} + > /dev/null
  prewarm=$(echo "$(date +%s.%N) - $t0" | bc)
  echo "[ltx2x4] $(ts) $cfg: page cache warmed in ${prewarm}s"
  t0=$(date +%s.%N)
  if step "ltx_2_${cfg}_nopreshard_warm2" "$LOG/ltx_2_${cfg}_nopreshard_warm2.log" bash -c "$GEN"; then
    wall=$(echo "$(date +%s.%N) - $t0" | bc)
    "$PY" - "$out" "$cfg" "$wall" "$prewarm" "$LOG/ltx_2_${cfg}_nopreshard_warm2.log" <<'EOF'
import json, re, sys
out, cfg, wall, prewarm, log = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
text = open(log).read()
gemma = re.findall(r"Loading checkpoint shards: 100%\|[^\]]*\[(\d+:\d+)<", text)
json.dump({"what": f"LTX-2 480x704x49 tp4 ({cfg}) standalone, weights sharded at load; the model's "
                   "files read into the page cache right before the run (warm by definition)",
           "warm_s": round(wall, 3), "prewarm_s": round(prewarm, 3),
           "host_text_encoder_load": gemma,
           "sharding_on_load_lines": len(re.findall(r"Sharding weights on load", text))},
          open(out, "w"), indent=2)
print(open(out).read())
EOF
  fi
  rm -rf "${NOPRE:?}"
done
echo "[ltx2x4] $(ts) ALL_DONE"
