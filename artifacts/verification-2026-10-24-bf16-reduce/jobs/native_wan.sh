#!/usr/bin/env bash
# Task 13: Wan2.1 on the PyTorch-native backend with the branch's own driver
# (the Wan2.1 report's methodology), guidance 5.0 only; the W4 reference
# hashes do not exist on this host, so --no-w4-check.
source /home/ubuntu/campaign/bf16-reduce/jobs/env.sh
NS=/home/ubuntu/campaign/native_src
NV=/home/ubuntu/venvs/native-312
out=$CAMPAIGN/runs/native_wan21
mkdir -p "$out"
cd "$NS"
HF_HUB_OFFLINE=1 PYTHONSAFEPATH=1 PYTHONPATH=$NS PATH=$NV/bin:$PATH \
  "$NV/bin/python" -m benchmark.neuron_cold_warm --record-dir "$out/records" \
    --guidance 5.0 --no-w4-check --out-json "$out/wan_2_1_neuron.json" --out-md "$out/wan_2_1_neuron.md" \
    > "$out/driver.log" 2>&1
rc=$?
echo "native_wan21_driver rc=$rc" | tee -a "$CAMPAIGN/timings.txt"
[ $rc -eq 0 ] && touch "$out/measure.done"
exit $rc
