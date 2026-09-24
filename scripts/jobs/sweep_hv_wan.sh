#!/usr/bin/env bash
# Campaign job: the step-caching sweep for HunyuanVideo and Wan 2.1 at their
# official step counts (50 each), every component on device.
#
#   scripts/supervise.sh sweep_hv_wan scripts/jobs/sweep_hv_wan.sh
#
# Requires the compile job to have finished; `difflet generate` refuses to run
# without artifacts, so a missing compile fails fast rather than silently
# measuring something else.
set -uo pipefail

ROOT=/home/ubuntu/Difflet
cd "${ROOT}"

export DIFFLET_RERUN_REPEATS="${DIFFLET_RERUN_REPEATS:-3}"
# This round: the caching-off reference plus the two probe-free modes. Calibrated
# adaptive is deferred — no calibration JSON exists yet, and fitting one for Wan
# needs a pairs collector that calibrate_teacache.py only has for HunyuanVideo.
export DIFFLET_RERUN_MODES="${DIFFLET_RERUN_MODES:-off cadence2 online}"

for model in hunyuan wan; do
  echo "########## sweep ${model} ########## $(date -Is)"
  bash scripts/rerun_caching_official_steps.sh "${model}"
  echo "[sweep] ${model} done rc=$?"
done

echo "########## collecting ########## $(date -Is)"
.venv/bin/python scripts/collect_caching_results.py \
  --root cclogs/caching-official-steps \
  --out cclogs/caching-official-steps/results.json
