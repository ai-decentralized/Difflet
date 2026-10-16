#!/usr/bin/env bash
# Phase 0 of docs/plans/2026-10-16-cuda-backend-support.md: record the GPU box's
# facts before any code depends on them. Prints everything the status log needs.
# Usage: scripts/cuda_env_check.sh [python]   (default: <repo>/.venv/bin/python, else python3)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-${PYTHON_BIN:-$ROOT/.venv/bin/python}}"
[ -x "$PY" ] || PY="$(command -v python3)"

echo "== host =="
echo "hostname: $(hostname)"
echo "os: $( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -a)"
echo "kernel: $(uname -r)"
echo "cpu: $(nproc 2>/dev/null || sysctl -n hw.ncpu) cores"
echo "ram: $(free -g 2>/dev/null | awk '/Mem:/{print $2" GB"}')"
echo "disk free: $(df -h "$ROOT" | awk 'NR==2{print $4" of "$2}')"
echo "hf cache: ${HF_HOME:-$HOME/.cache/huggingface}"

echo "== nvidia =="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
  nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | head -1
else
  echo "nvidia-smi: not found"
fi

echo "== python / torch =="
echo "python: $("$PY" --version 2>&1) ($PY)"
"$PY" - <<'EOF'
import importlib.util, torch
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda, "| cudnn:", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0),
          "| memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print("bf16 matmul: ok", tuple(y.shape))
for pkg in ("diffusers", "transformers", "accelerate", "safetensors", "triton", "flash_attn"):
    spec = importlib.util.find_spec(pkg)
    ver = None
    if spec is not None:
        try:
            ver = importlib.import_module(pkg).__version__
        except Exception:
            ver = "present"
    print(f"{pkg}: {ver}")
EOF
