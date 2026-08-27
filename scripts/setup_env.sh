#!/usr/bin/env bash
# Bootstrap the Difflet Neuron environment at <repo>/.venv.
#
# Recent Neuron DLAMI releases no longer ship the prebuilt
# /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference venv, so this script rebuilds
# the equivalent environment from requirements-neuron.lock. Exact pins matter
# beyond reproducibility: the AOT compile-cache key (~/.cache/difflet) includes
# the Python minor version and the torch / neuronx-cc / neuronx-distributed /
# diffusers / transformers versions, so hosts with identical envs can share
# compiled artifacts.
#
# Usage:
#   ./scripts/setup_env.sh          # creates <repo>/.venv
#   DIFFLET_VENV=/path ./scripts/setup_env.sh   # custom location
#
# To regenerate the lock after changing the toolchain:
#   .venv/bin/pip freeze --exclude-editable > requirements-neuron.lock
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${DIFFLET_VENV:-${ROOT}/.venv}"
PYTHON="${PYTHON:-python3}"
NEURON_INDEX="https://pip.repos.neuron.amazonaws.com"

ver="$("${PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "${ver}" in
  3.[0-9]) echo "error: Python >= 3.10 required (got ${ver})" >&2; exit 1 ;;
  3.1[0-9]) ;;
  *) echo "error: unsupported Python version ${ver}" >&2; exit 1 ;;
esac
if [[ "${ver}" != "3.12" ]]; then
  echo "warning: the reference environment is Python 3.12; ${ver} works but" \
    "cannot share compile caches (Python minor is part of the cache key)" >&2
fi

"${PYTHON}" -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install --extra-index-url "${NEURON_INDEX}" \
  -r "${ROOT}/requirements-neuron.lock"
# --no-deps: every dependency (incl. [test] extras) is already pinned by the
# lock. Letting pip re-resolve here can downgrade Neuron wheels (see the
# dependency comment in pyproject.toml).
"${VENV}/bin/pip" install -e "${ROOT}[test]" --no-deps

echo
echo "Environment ready at ${VENV}. Next steps:"
echo "  source ${VENV}/bin/activate"
echo "  huggingface-cli login   # FLUX.1-dev is a gated repo"
