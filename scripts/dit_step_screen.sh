#!/usr/bin/env bash
# One full-model DiT-step measurement through the production CLI: difflet compile, then two
# difflet generate runs (run 0 cold, run 1 warm), per-step DiT times parsed from the generate
# log (benchmark.adapters.trainium.parse_dit_step_seconds; FLUX has no per-step line, so its
# e2e wall and it/s are what you get).
#
# Usage:
#   scripts/dit_step_screen.sh <name> [KEY=VALUE ...] -- <difflet flags...>
#
#   KEY=VALUE pairs are exported (NEURON_RT_VIRTUAL_CORE_SIZE=2, DIFFLET_STRIDED_DMA=1, ...).
#   <difflet flags> go to both compile and generate: --model-id, --revision, --tp-degree, shape,
#   --quant/--quant-calibration, --host-vae, --cache-dir ... ; generate also gets --prompt,
#   --steps/--guidance-scale from STEPS / GUIDANCE (env, default 20 / 1.0), --seed 42.
#
# Environment: OUT_DIR (default artifacts/verification-<today>/model-screen), OUT_EXT (mp4|png,
# default mp4), STEPS, GUIDANCE, PROMPT, RUNS (default 2), PYTHON.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="${1:?usage: dit_step_screen.sh <name> [KEY=VALUE ...] -- <difflet flags>}"; shift
while [[ $# -gt 0 && "$1" != "--" ]]; do export "$1"; shift; done
[[ "${1:-}" == "--" ]] && shift
flags=("$@")

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then PYTHON="${ROOT}/.venv/bin/python"
  else PYTHON="/home/ubuntu/Difflet/.venv/bin/python"; fi
fi
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${ROOT}"
today="$(date -u +%F)"
OUT_DIR="${OUT_DIR:-${ROOT}/artifacts/verification-${today}/model-screen}"
OUT_EXT="${OUT_EXT:-mp4}"
STEPS="${STEPS:-20}"; GUIDANCE="${GUIDANCE:-1.0}"; RUNS="${RUNS:-2}"
PROMPT="${PROMPT:-a cinematic shot of a red fox running through a snowy forest}"
d="${OUT_DIR}/${name}"; mkdir -p "${d}/logs"
# private compiler scratch (the vendor ModelBuilder rmtree's its workdir; never share /tmp/nxd_model)
export BASE_COMPILE_WORK_DIR="${d}/nxd_scratch/"

echo "[screen] ${name}: $(env | grep -E '^(DIFFLET_|NEURON_RT_)' | sort | tr '\n' ' ')" | tee "${d}/screen.log"
echo "[screen] flags: ${flags[*]}" | tee -a "${d}/screen.log"
t0=$(date +%s)
"${PYTHON}" -m difflet.cli.main compile "${flags[@]}" > "${d}/logs/compile.log" 2>&1
rc=$?
compile_s=$(( $(date +%s) - t0 ))
echo "[screen] compile exit ${rc} in ${compile_s} s" | tee -a "${d}/screen.log"
[[ ${rc} -ne 0 ]] && { grep -E 'Error|error\]' "${d}/logs/compile.log" | tail -3 | cut -c1-200 | tee -a "${d}/screen.log"; }

walls=()
for ((r=0; r<RUNS; r++)); do
  t0=$(date +%s)
  "${PYTHON}" -m difflet.cli.main generate "${flags[@]}" --prompt "${PROMPT}" --steps "${STEPS}" \
    --guidance-scale "${GUIDANCE}" --seed 42 --output "${d}/run${r}.${OUT_EXT}" \
    --work-dir "${d}/work_run${r}" --keep-work-dir > "${d}/logs/generate_run${r}.log" 2>&1
  grc=$?
  walls+=($(( $(date +%s) - t0 )))
  echo "[screen] generate run ${r} exit ${grc} in ${walls[-1]} s" | tee -a "${d}/screen.log"
  [[ ${grc} -ne 0 ]] && { grep -E 'Error|error\]' "${d}/logs/generate_run${r}.log" | tail -3 | cut -c1-200 | tee -a "${d}/screen.log"; }
done

"${PYTHON}" - "${d}" "${name}" "${compile_s}" "${walls[*]}" "${flags[*]}" <<'PY'
import json, statistics, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]) if "__file__" in dir() else ".")
from benchmark.adapters.trainium import parse_dit_step_seconds
d, name, compile_s, walls, flags = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4].split(), sys.argv[5]
out = {"name": name, "flags": flags, "compile_s": compile_s, "runs": []}
for i, w in enumerate(walls):
    text = (d / "logs" / f"generate_run{i}.log").read_text(errors="replace")
    steps = parse_dit_step_seconds(text)
    rate = re.findall(r"([0-9.]+)\s*it/s", text)
    out["runs"].append({"run": i, "e2e_wall_s": int(w),
                        "dit_step_ms_median": round(statistics.median(steps) * 1000, 1) if steps else None,
                        "dit_step_ms_mean": round(statistics.fmean(steps) * 1000, 1) if steps else None,
                        "n_steps": len(steps), "it_per_s": float(rate[-1]) if rate else None})
(d / "result.json").write_text(json.dumps(out, indent=1))
print("[screen]", json.dumps({"name": name, "compile_s": compile_s,
                              "runs": [(r["e2e_wall_s"], r["dit_step_ms_median"], r["it_per_s"]) for r in out["runs"]]}))
PY
