#!/usr/bin/env bash
# Build the Difflet Neuron venv (SDK 2.30). One script, run it, done.
#   bash setup_neuron-2_30.sh [TARGET_DIR]   (default: /home/ubuntu/venvs/neuron-2_30)
set -euo pipefail

VENV="${1:-/home/ubuntu/venvs/neuron-2_30}"
NEURON_INDEX="https://pip.repos.neuron.amazonaws.com"

[[ -e "$VENV" ]] && { echo "ERROR: $VENV exists; remove it or pass another dir." >&2; exit 1; }

/usr/bin/python3.12 -m venv "$VENV"          # Neuron wheels are cp312 -> must be 3.12
source "$VENV/bin/activate"
pip install -U pip setuptools wheel

# Neuron 2.30 stack (only on the Neuron index) — pulls torch 2.9.1, torch-xla, etc.
pip install --extra-index-url "$NEURON_INDEX" \
    neuronx-cc==2.25.3371.0 nki==0.4.0 torch-neuronx==2.9.0.2.14.27725 \
    neuronx_distributed==0.19.28093 neuronx_distributed_inference==0.10.17970 \
    transformers==4.57.6

# Difflet runtime + test deps (from PyPI)
pip install \
    diffusers==0.38.0 accelerate==1.14.0 safetensors==0.8.0 sentencepiece==0.2.1 \
    einops==0.8.2 av==17.1.0 pillow==12.2.0 numpy==2.4.6 \
    pytest==9.1.1 pytest-xdist==3.8.0 expecttest==0.3.0 \
    imageio==2.37.3 imageio-ffmpeg==0.6.0 duckdb==1.5.4 ijson==3.5.0

python -c "import torch, torch_xla, neuronx_distributed, diffusers, transformers; \
print('OK', torch.__version__, '| diffusers', diffusers.__version__)"

cat <<EOF

DONE -> $VENV
Run Difflet (checkout is not pip-installed, use PYTHONPATH):
  export PATH=$VENV/bin:/opt/aws/neuron/bin:\$PATH
  cd /home/ubuntu/Difflet && PYTHONPATH=. pytest tests/unit -q
EOF
