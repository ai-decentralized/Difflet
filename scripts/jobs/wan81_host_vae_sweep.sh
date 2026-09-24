#!/usr/bin/env bash
# Wan 2.1 caching sweep at 81 frames, 50 steps, with the VAE decoded on the host.
#
#   scripts/supervise.sh wan81_host scripts/jobs/wan81_host_vae_sweep.sh
#
# Why the host VAE: the decoder's traced graph exceeds neuronx-cc's
# 5,000,000-instruction ceiling past 3 latent frames (NCC_EBVF030), and the ways
# around it are all closed on this stack, measured 2026-09-24:
#   - chunked AOT compile works and is bit-identical, but costs ~111 min per
#     graph and over 120 GB of compiler memory
#   - torch_xla eager compiles every operator separately; a toy decoder did not
#     finish in 900 s
#   - torch.compile(backend="openxla") fails in Dynamo's fake-tensor propagation
#     on silu ("Expected all tensors to be XLA tensors")
#   - torch_xla.compile reaches the compiler and hits an internal error,
#     NCC_INLA001 on a concatenate -- a compiler bug, not a size limit
#
# What this does and does not affect: step caching acts inside the denoise loop,
# and VAE decode runs once per generation outside it. So "skipped", the
# per-step loop time and PSNR are unaffected -- PSNR is a like-for-like
# comparison of cached against uncached output, both decoded the same way. Only
# the end-to-end column carries the host decode, and must say so.
#
# The transformer and UMT5 remain on device at 480x832x81.
set -uo pipefail

ROOT=/home/ubuntu/Difflet
cd "${ROOT}"

export DIFFLET_WAN_FRAMES=81
export DIFFLET_RERUN_REPEATS="${DIFFLET_RERUN_REPEATS:-3}"
export DIFFLET_RERUN_MODES="${DIFFLET_RERUN_MODES:-off cadence2 online}"
export DIFFLET_WAN_HOST_VAE=1

echo "########## wan 81f host-VAE sweep ########## $(date -Is)"
bash scripts/rerun_caching_official_steps.sh wan
echo "[wan81-host] sweep rc=$?"

echo "########## collecting ########## $(date -Is)"
.venv/bin/python scripts/collect_caching_results.py \
  --root cclogs/caching-official-steps \
  --out cclogs/caching-official-steps/results.json --latex
