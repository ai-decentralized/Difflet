#!/usr/bin/env bash
# `difflet serve` on one GPU, end to end (plan F3b; Flux and Wan F3c): start the server, wait for
# /ready, fetch two outputs (the second proves the model stays resident) — images over
# /v1/chat/completions, a Wan clip over /v1/videos/sync.
# Logs to /tmp/logs/cuda-serve-smoke-<ts>.log; timings in serve_timings.json next to --output.
#   scripts/cuda_serve_smoke.sh --output /tmp/qwen-serve.png [--steps 50 --seed 0 --port 8091]
#   scripts/cuda_serve_smoke.sh --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --output /tmp/wan.mp4
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
