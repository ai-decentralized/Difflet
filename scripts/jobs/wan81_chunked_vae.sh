#!/usr/bin/env bash
# Wan 2.1 at 81 frames with every component on device, using the chunked VAE.
#
#   scripts/supervise.sh wan81 scripts/jobs/wan81_chunked_vae.sh
#
# The transformer and UMT5 are already compiled at 480x832x81 and are reused --
# only the VAE is rebuilt, now as 7 chunk graphs of 3 latent frames each instead
# of one 21-frame graph that neuronx-cc rejects (NCC_EBVF030, 39,093,968
# instructions against a 5,000,000 ceiling). The split is bit-identical: the
# decoder's loop was already per-latent-frame, and its causal cache now lives in
# HBM as aliased state rather than being rebuilt inside one flattened graph.
set -uo pipefail

ROOT=/home/ubuntu/Difflet
export PATH="${ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${ROOT}"
export DIFFLET_BACKEND=trainium
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_NUM_CORES=4
cd "${ROOT}"

FRAMES=81

echo "===== [wan81] compile, chunked VAE ===== $(date -Is)"
t0=$(date +%s)
difflet compile \
  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --height 480 --width 832 --num-frames "${FRAMES}"
rc=$?
t1=$(date +%s)
echo "[wan81] compile exit=${rc} elapsed=$(( (t1 - t0) / 60 ))min"
if [ "${rc}" -ne 0 ]; then
  echo "[wan81] compile failed; not sweeping"
  grep -E "NCC_|Traceback|Error" /tmp/logs/wan81.log 2>/dev/null | tail -5
  exit "${rc}"
fi

echo "===== [wan81] artifacts ====="
difflet cache ls

echo "===== [wan81] caching sweep, 81 frames, 50 steps ===== $(date -Is)"
DIFFLET_WAN_FRAMES="${FRAMES}" bash scripts/rerun_caching_official_steps.sh wan
echo "[wan81] sweep rc=$?"

echo "===== [wan81] results ===== $(date -Is)"
.venv/bin/python scripts/collect_caching_results.py \
  --root cclogs/caching-official-steps \
  --out cclogs/caching-official-steps/results.json --latex
