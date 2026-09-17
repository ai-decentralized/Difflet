#!/usr/bin/env bash
# Same-host tp4 real-loop check for the tp4tcad campaign (2026-09-17): the
# committed tp4 baselines were measured on another host, so re-measure the
# plain tp4 real-loop (DiT call ms, loop ms/step) here for each model WITHOUT
# touching benchmark/trn2/<slug>.json -- step_realloop patches the JSON under
# benchmark/$DIFFLET_BENCH_DEVICE/, so the committed file is copied to a
# scratch device dir first and patched there. Run with the device idle.
#
#   benchmark/trn2/tp4_samehost_check.sh [model ...]   -> benchmark/trn2check/<slug>.json
set -u
MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(flux_1_dev qwen_image ltx_2 hunyuan_video wan_2_1)
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export DIFFLET_VENV="${DIFFLET_VENV:-$ROOT/.venv}"
export PATH="$DIFFLET_VENV/bin:$PATH"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
OUT=benchmark/trn2check
mkdir -p "$OUT/logs"
ts() { date -u +%FT%TZ; }
for m in "${MODELS[@]}"; do
  if python - "$m" <<'PY'
import json, sys
from pathlib import Path
p = Path("benchmark/trn2check") / f"{sys.argv[1]}.json"
sys.exit(0 if p.exists() and (json.loads(p.read_text()).get("samehost_check") or {}).get("done") else 1)
PY
  then echo "[check] $m already measured, skipping"; continue; fi
  cp "benchmark/trn2/$m.json" "$OUT/$m.json"
  echo "[check] $(ts) $m: tp4 real-loop on this host"
  if DIFFLET_BENCH_DEVICE=trn2check python -m benchmark.step_realloop --model "$m" --config tp4 \
       > "$OUT/logs/${m}_realloop.log" 2>&1; then
    python - "$m" <<'PY'
import datetime, json, sys
from pathlib import Path
p = Path("benchmark/trn2check") / f"{sys.argv[1]}.json"
d = json.loads(p.read_text())
d["samehost_check"] = {"done": True, "host": "ip-172-31-36-137 (tp4tcad campaign host, 2026-09-17)",
                       "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "note": "plain tp4 real-loop re-measured here; the committed benchmark/trn2/<slug>.json "
                               "baseline (other host) is untouched -- compare step_latency / loop_step_ms"}
p.write_text(json.dumps(d, indent=2))
st = d["step_latency"]
print(f"[check] {sys.argv[1]}: DiT call {st['mean']*1000:.1f} ms (n={st['n']}), loop {d.get('loop_step_ms')} ms/step")
PY
  else
    echo "[check] $(ts) $m FAILED; tail:"; tail -n 12 "$OUT/logs/${m}_realloop.log" | sed 's/^/    /'
  fi
done
echo "[check] $(ts) ALL_DONE"
