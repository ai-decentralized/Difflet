#!/usr/bin/env bash
# Qwen-Image on one GPU through the CUDA backend; logs to /tmp/logs/cuda-smoke-<ts>.log.
# Usage: scripts/cuda_smoke.sh [cuda_smoke.py args...]   e.g. --steps 4 --output /tmp/qwen4.png
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
"$PY" scripts/cuda_smoke.py "$@" 2>&1 | tee "$LOG"
