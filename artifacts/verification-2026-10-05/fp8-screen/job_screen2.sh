#!/usr/bin/env bash
# Follow-up screen after the A/B: sequence padding to 256 / 512 (pad128 was +15 % slower with or
# without the attention bounds, i.e. the 37 x 128 row tiling is worse than 12 x 390; a 512-multiple
# is the double-row-friendly tile the trace saw on the one layer pair that got the 2x).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export SCREEN_DIR="${ROOT}/artifacts/verification-2026-10-05/fp8-screen/screen"
S="${ROOT}/scripts/ptq_fp8_screen.sh"
while pgrep -f job_ab.sh >/dev/null; do sleep 60; done
run() { echo "=== $(date -u +%T) $*"; "$S" "$@" || echo "!!! $1 failed"; }
run pad512 --only both DIFFLET_WAN_PAD_TOKENS=512
run pad256 --only both DIFFLET_WAN_PAD_TOKENS=256
run best_a_pad512 --only both NEURON_RT_VIRTUAL_CORE_SIZE=2 DIFFLET_WAN_TENSORIZER_EXTRA=--vectorize-strided-dma DIFFLET_WAN_PAD_TOKENS=512
echo "=== $(date -u +%T) queue2 done"
