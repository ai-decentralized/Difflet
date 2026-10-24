#!/usr/bin/env bash
# Fallback PyTorch-native env (Task 13): copy of the /opt torch-neuronx 2.14 venv
# (Python 3.12, editable torch-neuronx 2.14.3) plus the branch's HF-stack pins,
# installed under a constraints file freezing the toolchain (as
# setup_native_env.sh does). The branch's own script needs an ECR DLC image and
# AWS credentials, neither of which exists on this host.
set -eu
SRC=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
DST=/home/ubuntu/venvs/native-312
LOG=/home/ubuntu/campaign/bf16-reduce/jobs/build_native_env.log
if [ ! -x "$DST/bin/python" ]; then
  cp -a "$SRC" "$DST"
  sed -i "s#$SRC#$DST#g" "$DST/bin/activate"
fi
cons=$(mktemp)
"$DST/bin/python" -m pip freeze --exclude-editable | grep -iE '^(torch|neuronx-cc|nki|numpy)( |==|@)' > "$cons"
cat "$cons"
"$DST/bin/python" -m pip install --only-binary=:all: -c "$cons" \
  diffusers==0.38.0 transformers==4.57.6 safetensors==0.8.0 tokenizers==0.22.2 \
  sentencepiece==0.2.2 huggingface_hub==0.36.2 accelerate==1.14.0 av==18.1.0 \
  imageio==2.37.4 imageio-ffmpeg==0.6.0
rm -f "$cons"
PATH="$DST/bin:$PATH" "$DST/bin/python" - <<'PY'
import importlib.metadata as md, sys
import torch, torch_neuronx  # noqa: F401
print("python", sys.version.split()[0])
for p in ("torch", "torch-neuronx", "neuronx-cc", "nki", "diffusers", "transformers"):
    print(f"  {p:14s} {md.version(p)}")
PY
