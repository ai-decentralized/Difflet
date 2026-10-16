#!/usr/bin/env bash
# Build the TorchNeuron (PyTorch-native, DIFFLET_BACKEND=neuron) environment on a
# Trainium host from the pre-release DLC: the venv requirements-neuron-native.lock
# describes, at ~/workspace/native_venv.
#
# Idempotent: rerun it any time. Each stage is skipped when already done:
#   pull/extract  only when the DLC image digest changed (stamp: workspace/.dlc-digest)
#   runtime debs  only the .debs whose version differs from the installed one;
#                 the neuron kernel module is reloaded if the driver changed
#   venv          created once (uv-managed CPython 3.14), packages reinstalled
#                 when the DLC changed
#
# Prerequisite: AWS credentials that can pull from the DLC's ECR repo
# (`aws login --remote`, or an instance profile).
#
# Usage:
#   ./scripts/setup_native_env.sh                # full setup + device check
#   ./scripts/setup_native_env.sh --build-ext    # also build the C++ extension
#   source ~/workspace/native_env.sh             # then, in every new shell
#
# Overrides: DLC_IMAGE, NATIVE_WORKSPACE, PYTHON_VERSION.
set -euo pipefail

DLC_IMAGE="${DLC_IMAGE:-421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest}"
WS="${NATIVE_WORKSPACE:-$HOME/workspace}"
VENV="$WS/native_venv"
PY="${PYTHON_VERSION:-3.14}"
PYTAG="cp${PY/./}"
UV_VERSION=0.13.0
BUILD_EXT=0
[[ "${1:-}" == "--build-ext" ]] && BUILD_EXT=1

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# Docker group membership only applies after a re-login; fall back to sudo.
DOCKER=(docker)
docker info >/dev/null 2>&1 || DOCKER=(sudo docker)

log "preflight"
ls /dev/neuron0 >/dev/null 2>&1 || die "no /dev/neuron0: not a Trainium/Inferentia host?"
command -v docker >/dev/null || { sudo apt-get update && sudo apt-get install -y docker.io; }

if aws sts get-caller-identity >/dev/null 2>&1; then
  log "pulling $DLC_IMAGE"
  registry="${DLC_IMAGE%%/*}"
  region="$(cut -d. -f4 <<<"$registry")"
  aws ecr get-login-password --region "$region" \
    | "${DOCKER[@]}" login --username AWS --password-stdin "$registry" >/dev/null
  "${DOCKER[@]}" pull -q "$DLC_IMAGE"
elif "${DOCKER[@]}" image inspect "$DLC_IMAGE" >/dev/null 2>&1; then
  log "no AWS credentials: using the local $DLC_IMAGE (run 'aws login --remote' to check for a newer one)"
else
  die "no AWS credentials: run 'aws login --remote' (or attach an instance profile)"
fi
digest="$("${DOCKER[@]}" image inspect --format '{{.Id}}' "$DLC_IMAGE")"

fresh=0
if [[ "$(cat "$WS/.dlc-digest" 2>/dev/null)" == "$digest" ]]; then
  log "DLC unchanged ($digest), keeping $WS"
else
  log "extracting /workspace from the DLC"
  if [[ -d "$WS" ]]; then
    # Keep the old tree (it may hold local edits to torch_neuron_eager).
    bak="$WS.$(date +%Y%m%d-%H%M%S)"
    echo "moving the previous $WS to $bak"
    mv "$WS" "$bak"
  fi
  tmp="$(mktemp -d "$(dirname "$WS")/.dlc-extract.XXXX")"
  cid="$("${DOCKER[@]}" create "$DLC_IMAGE")"
  "${DOCKER[@]}" cp "$cid:/workspace" "$tmp/" && "${DOCKER[@]}" rm "$cid" >/dev/null
  sudo chown -R "$(id -u):$(id -g)" "$tmp"
  mv "$tmp/workspace" "$WS" && rmdir "$tmp"
  echo "$digest" > "$WS/.dlc-digest"
  fresh=1
fi

log "Neuron runtime packages"
need=()
for deb in "$WS"/runtime_artifacts/*.deb; do
  pkg="$(dpkg-deb -f "$deb" Package)"
  want="$(dpkg-deb -f "$deb" Version)"
  have="$(dpkg-query -W -f '${Version}' "$pkg" 2>/dev/null || true)"
  if [[ "$have" == "$want" ]]; then echo "  $pkg $want (installed)"
  else echo "  $pkg ${have:-none} -> $want"; need+=("$deb"); fi
done
if ((${#need[@]})); then
  sudo apt-get update -qq
  sudo apt-get install -y -qq dkms build-essential "linux-headers-$(uname -r)"
  driver_changed=0
  for deb in "${need[@]}"; do
    [[ "$(dpkg-deb -f "$deb" Package)" == aws-neuronx-dkms ]] && driver_changed=1
  done
  # Packages the DLAMI held at its own versions would otherwise block upgrades.
  sudo apt-mark unhold $(for d in "${need[@]}"; do dpkg-deb -f "$d" Package; done) >/dev/null || true
  sudo dpkg -i "${need[@]}"
  if ((driver_changed)); then
    log "reloading the neuron kernel module (new driver)"
    sudo rmmod neuron && sudo modprobe neuron \
      || die "could not reload the neuron module (a process holds the device?); reboot and rerun"
  fi
fi

log "venv $VENV (Python $PY)"
export PATH="$HOME/.local/bin:$PATH"
if ! uv --version 2>/dev/null | grep -q "$UV_VERSION"; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh -s -- --quiet
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --seed --python "$PY" "$VENV"
  fresh=1
fi
export UV_PROJECT_ENVIRONMENT="$VENV" VIRTUAL_ENV="$VENV"

if ((fresh)) || ! "$VENV/bin/python" -c 'import torch_neuronx' 2>/dev/null; then
  shopt -s nullglob
  wheels=("$WS"/nki_wheels/*-"$PYTAG"-*.whl "$WS"/neuronx_cc_wheels/*-"$PYTAG"-*.whl)
  shopt -u nullglob
  ((${#wheels[@]} == 2)) || die "expected one nki and one neuronx_cc $PYTAG wheel, found: ${wheels[*]:-none}
  (available: $(ls "$WS"/nki_wheels "$WS"/neuronx_cc_wheels | tr '\n' ' '))"
  uv pip install "${wheels[@]}"
  # Installs the matching torch and all its deps.
  (cd "$WS/torch_neuron_eager" && uv pip install -e '.[dev]')

  # Difflet's HF stack, under constraints so the resolver cannot replace the
  # toolchain (pins: requirements-neuron-native.lock).
  cons="$(mktemp)"
  "$VENV/bin/python" -m pip freeze --exclude-editable \
    | grep -iE '^(torch|neuronx-cc|nki|numpy)( |==|@)' > "$cons"
  uv pip install --only-binary=:all: -c "$cons" \
    diffusers==0.38.0 transformers==4.57.6 safetensors==0.8.0 tokenizers==0.22.2 \
    sentencepiece==0.2.2 huggingface_hub==0.36.2 accelerate==1.14.0 av==18.1.0 \
    imageio==2.37.4 imageio-ffmpeg==0.6.0
  rm -f "$cons"
else
  echo "packages already installed (DLC unchanged)"
fi

if ((BUILD_EXT)); then
  log "building the torch_neuron_eager C++ extension"
  (cd "$WS/torch_neuron_eager" && sudo ./install_bazelisk.sh \
    && USE_BAZEL=1 "$VENV/bin/python" setup.py build_ext)
fi

cat > "$WS/native_env.sh" <<EOF
# source this in every shell that runs the PyTorch-native neuron backend
source "$VENV/bin/activate"
export UV_PROJECT_ENVIRONMENT="$VENV"
EOF

log "verifying"
neuron-ls | head -12
# torch_neuronx spawns the neuronx-cc console script, so the venv's bin must be
# on PATH (as it is after `source native_env.sh`).
PATH="$VENV/bin:$PATH" "$VENV/bin/python" - <<'PY'
import importlib.metadata as md, sys
import torch, torch_neuronx  # noqa: F401
print("python", sys.version.split()[0])
for p in ("torch", "torch-neuronx", "neuronx-cc", "nki", "diffusers", "transformers"):
    print(f"  {p:14s} {md.version(p)}")
x = torch.arange(8, dtype=torch.float32, device="neuron")
y = (x * 2 + 1).cpu()
assert y.tolist() == [2 * i + 1 for i in range(8)], y
print("device op OK on", x.device)
PY
dpkg -l | awk '/aws-neuronx/ {print "  " $2, $3}'

log "done. In a new shell: source $WS/native_env.sh"
