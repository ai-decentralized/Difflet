#!/usr/bin/env bash
# `difflet serve` for Qwen-Image on one GPU, end to end (plan F3b): start the server, wait for
# /ready, fetch two images over /v1/chat/completions (the second proves the model stays resident).
# Logs to /tmp/logs/cuda-serve-smoke-<ts>.log; timings in serve_timings.json next to --output.
#   scripts/cuda_serve_smoke.sh --output /tmp/qwen-serve.png [--steps 50 --seed 0 --port 8091]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
mkdir -p /tmp/logs
LOG="/tmp/logs/cuda-serve-smoke-$(date +%Y%m%d-%H%M%S).log"
export DIFFLET_BACKEND=cuda
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$ROOT"
echo "log: $LOG"
"$PY" scripts/cuda_serve_smoke.py "$@" 2>&1 | tee "$LOG"
