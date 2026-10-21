#!/usr/bin/env bash
# Wan 2.2 T2V-A14B on one GPU through `difflet run` on the CUDA backend. GPU-only.
# Logs to /tmp/logs/cuda-smoke-wan-<ts>.log (CUDA_SMOKE_LOGDIR overrides the directory) and,
# after a generate, the timings to cuda-smoke-wan-<ts>.json next to it. Load / generate times
# come from the CLI's own "[difflet] loaded in" / "wrote ... in" lines; the peak device memory
# (both experts resident) from nvidia-smi sampled once a second into cuda-smoke-wan-<ts>.gpu.txt.
#   scripts/cuda_smoke_wan.sh --prompt "a cat" --output /tmp/wan.mp4 [--steps 40 --num-frames 9 --seed 0]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
LOGDIR="${CUDA_SMOKE_LOGDIR:-/tmp/logs}"

# parse_timings LOG JSON [difflet run args...]: the CLI's timing lines (and the JSON's
# .gpu.txt memory samples, MiB, when present) → a JSON; no file when the log has no generate
# line, e.g. --help.
parse_timings() {
  "$PY" - "$@" <<'PY'
import json, os, re, sys
log, out, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
samples = out[: -len(".json")] + ".gpu.txt"
text = open(log).read()
def flag(name, default=None, cast=str):
    return cast(argv[argv.index(name) + 1]) if name in argv[:-1] else default
loaded = re.search(r"^\[difflet\] loaded in ([0-9.]+) s", text, re.M)
wrote = re.search(r"^\[difflet\] wrote (\S+) .*in ([0-9.]+) s$", text, re.M)
if wrote is None:
    print("[smoke] no generate in the log; no timings written")
    sys.exit(0)
steps = flag("--steps", 40, int)  # WanCudaOrchestrator.DEFAULT_STEPS
timings = {
    "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "load_s": float(loaded.group(1)) if loaded else None,
    "generate_s": float(wrote.group(2)),
    "per_step_s": round(float(wrote.group(2)) / max(steps, 1), 2),
    "steps": steps,
    "num_frames": flag("--num-frames", 9, int),
    "seed": flag("--seed", 42, int),
    "output": wrote.group(1),
    "log": log,
}
if os.path.isfile(samples):
    used = [int(v) for v in open(samples).read().split() if v.isdigit()]
    timings["peak_gpu_mem_mib"] = max(used) if used else None
open(out, "w").write(json.dumps(timings, indent=2))
print("[smoke] " + json.dumps(timings))
PY
}

if [ "${1:-}" = "--parse-timings" ]; then
  shift
  parse_timings "$@"
  exit 0
fi

mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOGDIR/cuda-smoke-wan-$STAMP.log"
export DIFFLET_BACKEND=cuda
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$ROOT"
echo "log: $LOG"
SAMPLER=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
    > "$LOGDIR/cuda-smoke-wan-$STAMP.gpu.txt" 2>/dev/null &
  SAMPLER=$!
  trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT
fi
"$PY" -m difflet.cli.main run --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --tp-degree 1 "$@" 2>&1 | tee "$LOG"
[ -z "$SAMPLER" ] || kill "$SAMPLER" 2>/dev/null || true
parse_timings "$LOG" "$LOGDIR/cuda-smoke-wan-$STAMP.json" "$@"
