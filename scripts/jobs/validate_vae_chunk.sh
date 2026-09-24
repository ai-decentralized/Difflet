#!/usr/bin/env bash
# Does a 3-latent-frame Wan VAE graph compile at 480x832?
#
# That is the premise the chunked decode rests on: if one chunk compiles, 81
# frames can be decoded as 7 chunks with the causal cache held in HBM, all on
# device. If it does not, chunking at this chunk size buys nothing.
#
# Measured ceilings on this host, 2026-09-23 (neuronx-cc NCC_EBVF030, limit
# 5,000,000 instructions): 4 latent frames 6,675,705; 5 8,871,691; 6 11,164,146;
# 7 11,823,804; 9 15,719,922; 21 39,093,968 -- all rejected. 3 is the candidate.
#
#   scripts/supervise.sh validate_vae_chunk scripts/jobs/validate_vae_chunk.sh
set -uo pipefail

ROOT=/home/ubuntu/Difflet
export PATH="${ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${ROOT}"
export DIFFLET_BACKEND=trainium
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_NUM_CORES=4
cd "${ROOT}"

echo "===== [validate] Wan VAE, 9 frames = 3 latent frames, 480x832 ===== $(date -Is)"
t0=$(date +%s)
.venv/bin/python -m difflet.cli.stage \
  --orchestrator Wan-AI/Wan2.1-T2V-14B-Diffusers --stage vae --stage-mode compile \
  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --height 480 --width 832 --num-frames 9
rc=$?
t1=$(date +%s)
echo "[validate] exit=${rc} elapsed=$(( (t1 - t0) / 60 ))min"
exit "${rc}"
