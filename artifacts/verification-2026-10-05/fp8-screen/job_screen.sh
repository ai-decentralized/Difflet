#!/usr/bin/env bash
# fp8 step-time screen queue, 2026-10-05 (trn2.3xlarge, tp4, Wan 2.1 2 real blocks, static scales).
# Each line = one scripts/ptq_fp8_screen.sh call; results land in screen/<name>.json.
# Baseline on this host (idle): bf16 35.69 ms, fp8 static 32.34 ms (screen/base.json).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export SCREEN_DIR="${ROOT}/artifacts/verification-2026-10-05/fp8-screen/screen"
S="${ROOT}/scripts/ptq_fp8_screen.sh"
run() { echo "=== $(date -u +%T) $*"; "$S" "$@" || echo "!!! $1 failed"; }

# E9: sequence-level padding (double-row hypothesis). Both arms: bf16 may move too.
run pad128           --only both DIFFLET_WAN_PAD_TOKENS=128
run pad16            --only fp8  DIFFLET_WAN_PAD_TOKENS=16
# E2: bf16 dequant (unmeasured switch from 4dec3b2).
run bf16_dequant     --only fp8  DIFFLET_FP8_BF16_DEQUANT=1
# Reference winner of the 10-05 screen, re-measured on this host, and its bf16 control.
run best_a_bf16q     --only fp8  NEURON_RT_VIRTUAL_CORE_SIZE=2 DIFFLET_WAN_TENSORIZER_EXTRA=--vectorize-strided-dma DIFFLET_FP8_BF16_QUANT=1
run best_a           --only both NEURON_RT_VIRTUAL_CORE_SIZE=2 DIFFLET_WAN_TENSORIZER_EXTRA=--vectorize-strided-dma
# Stacking on the winner.
run best_a_bf16q_pad128 --only both NEURON_RT_VIRTUAL_CORE_SIZE=2 DIFFLET_WAN_TENSORIZER_EXTRA=--vectorize-strided-dma DIFFLET_FP8_BF16_QUANT=1 DIFFLET_WAN_PAD_TOKENS=128
run best_a_bf16q_dq  --only fp8  NEURON_RT_VIRTUAL_CORE_SIZE=2 DIFFLET_WAN_TENSORIZER_EXTRA=--vectorize-strided-dma DIFFLET_FP8_BF16_QUANT=1 DIFFLET_FP8_BF16_DEQUANT=1
# E6: compiler flags (2.26 libwalrus strings), each alone on the shipped fp8 config.
run flag_mm_remat    --only fp8  DIFFLET_WAN_CC_EXTRA=--enable-mm-transpose-remat-optimization
run flag_lnc_single  --only fp8  DIFFLET_WAN_CC_EXTRA=--enable-lnc-single-graph-compilation
run flag_fp8_nosat   --only fp8  DIFFLET_WAN_CC_EXTRA=--disable-fp8-saturation
run flag_mm_reorder  --only fp8  DIFFLET_WAN_CC_EXTRA=--enable-internal-postsched-mm-accum-reorder
echo "=== $(date -u +%T) queue done"
