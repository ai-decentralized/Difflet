#!/usr/bin/env bash
# Build a Neuron venv on a host that is NOT a Trainium EC2 instance.
#
# Why: difflet's unit suite is written against the Neuron toolchain, so off a
# Trainium host ~107 tests fail and ~29 modules fail to collect purely on
# missing imports. That hides real regressions and blocks any change to the
# Trainium code path. With this setup the same suite runs 2090 passed / 3
# failed (the 3 are pre-existing and unrelated).
#
# This gives IMPORT-level and CPU-level coverage only. There is no Neuron
# driver and no device: anything that actually executes on hardware still
# cannot run here. Treat a green run as "did not break the Trainium code
# path", never as "verified on Trainium".
#
# Four things block the toolchain off-host, each with a supported escape:
#   1. libtorchneuron.so needs libnrt.so.1, which ships in an apt package —
#      extracted from the .deb here, no root and no system changes.
#   2. torch_xla shells out to `libneuronpjrt-path`, a console script in the
#      venv — so the venv's bin must be on PATH.
#   3. libneuronxla hard-codes /opt/aws/neuron/lib/libnrt.so.1 and checks it
#      exists — NEURON_INTERNAL_SKIP_LIBNRT_CHECK bypasses that.
#   4. torch_neuronx reads DMI product_name and refuses on non-EC2
#      ("Unsupported Platform - Google Compute Engine") —
#      NEURON_PLATFORM_TARGET_OVERRIDE names the target instead.
#
# Usage:
#   ./scripts/setup_neuron_venv_offhost.sh /path/to/venv-parent
#   source /path/to/venv-parent/neuron_env.sh
#   python -m pytest tests/unit -q

set -euo pipefail

DEST="${1:?usage: $0 <directory to create the venv in>}"
VENV="$DEST/nrnenv"
LIBDIR="$DEST/nrnlib"
NEURON_PIP="https://pip.repos.neuron.amazonaws.com"
NEURON_APT="https://apt.repos.neuron.amazonaws.com"
TARGET="${NEURON_TARGET:-trn2}"

mkdir -p "$DEST"

echo "==> creating venv at $VENV"
# python3-venv is often absent; virtualenv needs no system package.
python -m virtualenv "$VENV" >/dev/null 2>&1 || python -m venv "$VENV"

echo "==> installing torch (CPU) + the Neuron toolchain"
"$VENV/bin/pip" install -q torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install -q numpy neuronx-distributed --extra-index-url "$NEURON_PIP"

echo "==> installing difflet's runtime deps"
# transformers must stay on 4.x: neuronx_distributed imports
# transformers.utils.fx, which 5.x removed (106 failures without this pin).
"$VENV/bin/pip" install -q pytest "diffusers==0.38.0" "transformers<5" accelerate \
    pydantic fastapi uvicorn python-dotenv httpx python-multipart av \
    imageio-ffmpeg safetensors

echo "==> extracting libnrt from the Neuron apt package (no root needed)"
DEB=$(curl -s "$NEURON_APT/dists/jammy/main/binary-amd64/Packages" \
      | grep -E '^Filename:.*aws-neuronx-runtime-lib' | sort -V | tail -1 | awk '{print $2}')
[ -n "$DEB" ] || { echo "could not find aws-neuronx-runtime-lib in the apt index" >&2; exit 1; }
mkdir -p "$LIBDIR"
curl -s -o "$DEST/runtime-lib.deb" "$NEURON_APT/$DEB"
dpkg-deb -x "$DEST/runtime-lib.deb" "$LIBDIR"
rm -f "$DEST/runtime-lib.deb"

cat > "$DEST/neuron_env.sh" <<EOF
# source this before running the suite
export PATH="$VENV/bin:\$PATH"
export LD_LIBRARY_PATH="$LIBDIR/opt/aws/neuron/lib:\${LD_LIBRARY_PATH:-}"
export NEURON_INTERNAL_SKIP_LIBNRT_CHECK=1
export NEURON_PLATFORM_TARGET_OVERRIDE=$TARGET
EOF

echo "==> verifying imports"
# shellcheck disable=SC1090
source "$DEST/neuron_env.sh"
"$VENV/bin/python" - <<'PY'
mods = ["torch_xla", "torch_neuronx", "neuronx_distributed",
        "neuronx_distributed.parallel_layers.layers",
        "neuronx_distributed.trace.hlo_utils", "nki", "nkilib"]
bad = []
for m in mods:
    try:
        __import__(m)
    except Exception as exc:
        bad.append(f"{m}: {type(exc).__name__}: {exc}")
if bad:
    raise SystemExit("FAILED imports:\n  " + "\n  ".join(bad))
print("all Neuron imports OK")
PY

echo
echo "done. To use it:"
echo "    source $DEST/neuron_env.sh"
echo "    python -m pytest tests/unit -q"
