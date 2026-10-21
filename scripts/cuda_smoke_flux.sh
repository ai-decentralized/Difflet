#!/usr/bin/env bash
# FLUX.1-dev on one GPU through `difflet run` on the CUDA backend. GPU-only.
# Logs to /tmp/logs/cuda-smoke-flux-<ts>.log. Load / generate timings come from the CLI's
# own "[difflet] loaded in" / "wrote ... in" lines.
#   scripts/cuda_smoke_flux.sh --prompt "a cat" --output /tmp/flux.png [--steps 28 --seed 0 --guidance-scale 3.5]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
mkdir -p /tmp/logs
LOG="/tmp/logs/cuda-smoke-flux-$(date +%Y%m%d-%H%M%S).log"
export DIFFLET_BACKEND=cuda
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$ROOT"
echo "log: $LOG"
"$PY" -m difflet.cli.main run --model-id black-forest-labs/FLUX.1-dev --tp-degree 1 "$@" 2>&1 | tee "$LOG"
