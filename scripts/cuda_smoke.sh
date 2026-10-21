#!/usr/bin/env bash
# Qwen-Image on one GPU through `difflet run` on the CUDA backend.
# Logs to /tmp/logs/cuda-smoke-<ts>.log.
#   scripts/cuda_smoke.sh --prompt "a cat" --output /tmp/qwen.png [--steps 50 --seed 0]
# Per-phase timings (load / encode / generate, peak memory) come from scripts/cuda_smoke.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
mkdir -p /tmp/logs
LOG="/tmp/logs/cuda-smoke-$(date +%Y%m%d-%H%M%S).log"
export DIFFLET_BACKEND=cuda
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$ROOT"
echo "log: $LOG"
"$PY" -m difflet.cli.main run --model-id Qwen/Qwen-Image --tp-degree 1 "$@" 2>&1 | tee "$LOG"
