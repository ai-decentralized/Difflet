#!/usr/bin/env bash
# Wan 2.1 81-frame rows re-measured after 26c0ffd (transformer traced with
# NEURON_RT_VIRTUAL_CORE_SIZE=2 + --vectorize-strided-dma): the NKI attention now
# runs on both physical cores of each LNC2 core. One device, so every job runs
# serially, and each step is skipped when its result already exists.
#
#   1. tp4: compile, cold and warm standalone latency, DiT step; then the final
#      latents against the single-core graph's (same prompt, seed 42).
#   2. tp4+sp, then tp2cp2 (ulysses; it ran out of HBM on the single-core graph).
#
# Results: benchmark/trn2-eval-vc2/<slug>_<config>.json. The single-core graph's
# rows stay in benchmark/trn2-eval/ for the before/after comparison.
#
#   setsid nohup benchmark/trn2/eval/queue_wan_vc2.sh > LOG 2>&1 &   (venv active)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFLET_BENCH_DEVICE=trn2-eval-vc2
PY="${PYTHON_BIN:-python}"
RES="benchmark/$DIFFLET_BENCH_DEVICE"
LOG="$RES/logs"
NUM=benchmark/trn2/eval/vc2
SINGLE_CORE_LATENTS="${SINGLE_CORE_LATENTS:-$HOME/work/scratch/numerics}"
mkdir -p "$LOG" "$NUM"
ts() { date -u +%FT%TZ; }
has() { [[ -f "$RES/$1.json" ]] && "$PY" -c "import json,sys; d=json.load(open('$RES/$1.json')); sys.exit(0 if d.get('$2') not in (None, {}, []) else 1)"; }
step() {  # step <name> <log> <cmd...>
  local name="$1" log="$2"; shift 2
  echo "[queue] $(ts) $name start -> $log"
  local t0=$SECONDS
  if "$@" >"$log" 2>&1; then echo "[queue] $(ts) $name done in $((SECONDS - t0))s"; return 0; fi
  echo "[queue] $(ts) $name FAILED after $((SECONDS - t0))s; tail:"; tail -n 20 "$log" | sed 's/^/    /'
  return 1
}
compare_latents() {  # compare_latents <config> <single-core latents> <out json>
  "$PY" - "$HOME/.cache/difflet/work/bench_$1/latents.pt" "$2" "$3" <<'EOF'
import json, sys, torch
new, old, out = sys.argv[1:4]
a = torch.load(new, map_location="cpu").float()
b = torch.load(old, map_location="cpu").float()
d = (a - b).abs()
mse = float(((a - b) ** 2).mean())
peak = float(b.abs().max())
doc = {"lnc2_latents": new, "single_core_latents": old, "shape": list(a.shape),
       "bitwise_equal": bool(torch.equal(a, b)), "max_abs_diff": float(d.max()),
       "mean_abs_diff": float(d.mean()), "ref_max_abs": peak,
       "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
       "psnr_db": (float("inf") if mse == 0 else float(10 * torch.log10(torch.tensor(peak ** 2 / mse))))}
json.dump(doc, open(out, "w"), indent=2)
print(json.dumps(doc, indent=2))
EOF
}

for cfg in tp4 tp4sp tp2cp2; do
  [[ "$cfg" == tp4 ]] && s=wan_2_1 || s="wan_2_1_$cfg"
  if ! has "$s" compile_seconds; then
    step "$s/compile" "$LOG/${s}_compile.log" \
      "$PY" -m benchmark.bench --model wan_2_1 --config "$cfg" --skip-download --compile-only || continue
  fi
  has "$s" e2e_warm || step "$s/cold_warm" "$LOG/${s}_cold_warm.log" \
    "$PY" -m benchmark.cold_warm_e2e --model wan_2_1 --config "$cfg"
  has "$s" step_latency || step "$s/step" "$LOG/${s}_realloop.log" \
    "$PY" -m benchmark.step_realloop --model wan_2_1 --config "$cfg"
  ref="$SINGLE_CORE_LATENTS/latents_81f_${cfg}_single_core.pt"
  if [[ -f "$ref" && ! -f "$NUM/${s}_latents_vs_single_core.json" ]]; then
    step "$s/numerics" "$LOG/${s}_numerics.log" compare_latents "$s" "$ref" "$NUM/${s}_latents_vs_single_core.json"
  fi
done
echo "[queue] $(ts) ALL_DONE"
