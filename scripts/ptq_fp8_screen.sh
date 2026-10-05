#!/usr/bin/env bash
# 2-block Wan 2.1 fp8 screen: one named configuration per invocation.
#
# Runs scripts/ptq_fp8_device_probe.py on the real Wan 2.1 transformer truncated to
# --num-layers blocks (tp4, 480x832x9, static scales) under the environment switches
# and compiler extras given on the command line, in its own work dir (the compile
# cache key does not see any DIFFLET_WAN_* / DIFFLET_FP8_* switch), and records the
# per-arm forward time into <screen-dir>/<name>.json.
#
# Usage:
#   scripts/ptq_fp8_screen.sh <name> [--only bf16|fp8|both] [KEY=VALUE ...]
#
#   KEY=VALUE pairs are exported for the probe (e.g. DIFFLET_FP8_BF16_QUANT=1,
#   NEURON_RT_VIRTUAL_CORE_SIZE=2, DIFFLET_WAN_TENSORIZER_EXTRA=--vectorize-strided-dma,
#   DIFFLET_WAN_CC_EXTRA=..., DIFFLET_WAN_OPT_LEVEL=-O2).
#
# Environment:
#   SCREEN_DIR     where <name>.json lands (default artifacts/verification-<today>/fp8-screen/screen)
#   WORK_ROOT      per-config work dirs (default $CLAUDE_JOB_DIR/tmp/screen or /tmp/difflet_fp8_screen)
#   MODEL_DIR      HF transformer/ dir (default: the Wan 2.1 snapshot in the HF cache)
#   TEXT_PT        optional real prompt embeddings (.pt); random text otherwise
#   NUM_LAYERS     default 2;  ITERS default 10;  TP default 4
#   PYTHON         interpreter (default <repo>/.venv/bin/python, else /home/ubuntu/Difflet/.venv)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="${1:?usage: ptq_fp8_screen.sh <name> [--only arm] [KEY=VALUE ...]}"; shift
only="both"
if [[ "${1:-}" == "--only" ]]; then only="$2"; shift 2; fi
for kv in "$@"; do export "$kv"; done

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then PYTHON="${ROOT}/.venv/bin/python"
  else PYTHON="/home/ubuntu/Difflet/.venv/bin/python"; fi
fi
# The Neuron runtime resolves libneuronpjrt-path from PATH (README: .venv/bin must be on it).
export PATH="$(dirname "${PYTHON}"):${PATH}"
today="$(date -u +%F)"
SCREEN_DIR="${SCREEN_DIR:-${ROOT}/artifacts/verification-${today}/fp8-screen/screen}"
WORK_ROOT="${WORK_ROOT:-${CLAUDE_JOB_DIR:+${CLAUDE_JOB_DIR}/tmp/screen}}"
WORK_ROOT="${WORK_ROOT:-/tmp/difflet_fp8_screen}"
if [[ -z "${MODEL_DIR:-}" ]]; then
  MODEL_DIR="$(ls -d "${HOME}"/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/*/transformer | head -1)"
fi
mkdir -p "${SCREEN_DIR}" "${WORK_ROOT}"
work="${WORK_ROOT}/${name}"

args=(--work-dir "${work}" --real-model-dir "${MODEL_DIR}" --num-layers "${NUM_LAYERS:-2}"
      --tp-degree "${TP:-4}" --static --height 480 --width 832 --num-frames 9
      --only "${only}" --iters "${ITERS:-10}" --force-clean)
[[ -n "${TEXT_PT:-}" ]] && args+=(--text-pt "${TEXT_PT}")

log="${SCREEN_DIR}/${name}.log"
echo "[screen] ${name}: only=${only} switches: $*" | tee "${log}"
{ env | grep -E '^(DIFFLET_|NEURON_RT_)' | sort || true; } | tee -a "${log}"
started=$(date +%s)
set +e
PYTHONPATH="${ROOT}" "${PYTHON}" "${ROOT}/scripts/ptq_fp8_device_probe.py" "${args[@]}" >> "${log}" 2>&1
status=$?
set -e
grep -E '^\[probe' "${log}" || true
echo "[screen] probe exit ${status}, wall $(( $(date +%s) - started )) s" | tee -a "${log}"

# Summarise: per arm forward ms (mean), compile s, nonfinite count, compiler args, switches.
"${PYTHON}" - "${work}/ptq_probe_report.json" "${SCREEN_DIR}/${name}.json" "$*" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
checks = report.get("checks", {})
out = {"switches": sys.argv[3], "passed": report.get("passed")}
for name, arm in report.get("arms", {}).items():
    fwd = arm.get("forward_ms") or {}
    out[name] = {
        "forward_ms": fwd.get("mean"), "forward_ms_median": fwd.get("median"),
        "compile_s": arm.get("compile_seconds"), "nonfinite": arm.get("output_nonfinite"),
        "cosine_vs_cpu": checks.get(f"device_{name}_vs_cpu_{name}", {}).get("cosine"),
        "compiler_args": arm.get("compiler_args"), "error": arm.get("error"),
    }
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: (v.get("forward_ms") if isinstance(v, dict) else v) for k, v in out.items()}))
PY
exit "${status}"
