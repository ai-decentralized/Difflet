#!/usr/bin/env bash
# fingerprint_logs.sh <compile log>...: all-reduce dtype counts of every HLO
# module each compile log names, from the neuronx-cc compile cache.
CACHE=$(ls -d /var/tmp/neuron-compile-cache/neuronxcc-*/ | head -1)
for log in "$@"; do
  for mod in $(grep -o 'MODULE_[0-9a-f]*+[0-9a-f]*' "$log" | sort -u); do
    printf '%s ' "$(basename "$log")"
    /home/ubuntu/venvs/difflet-neuron/bin/python /home/ubuntu/campaign/bf16-reduce/jobs/hlo_allreduce.py \
      "$CACHE$mod/model.hlo_module.pb" 2>/dev/null | sed "s#$CACHE##; s#/model.hlo_module.pb##"
  done
done
