#!/usr/bin/env bash
# Task 10 step 3: device profile of one DiT forward per (model, dtype).
# Four collective workers (one per logical NeuronCore, LNC=2) run the rank NEFF
# with random inputs; worker 0 is profiled. Summaries via neuron-explorer view.
#   profile.sh <label> <neff>
source /home/ubuntu/campaign/bf16-reduce/jobs/env.sh
label=$1 neff=$2
out=$CAMPAIGN/runs/profile/$label
mkdir -p "$out"
pids=()
for r in 0 1 2 3; do
  NEURON_RT_VISIBLE_CORES=$r neuron-explorer capture -n "$neff" -s "$out/rank$r.ntff" \
    --collectives-worker-start-id "$r" --collectives-worker-count 4 \
    --collectives-workers-per-node 4 --collectives-profile-id 0 \
    --num-exec 4 --profile-nth-exec 4 > "$out/capture_rank$r.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "capture rc=$rc"
for f in "$out"/rank0*.ntff; do
  [ -e "$f" ] || continue
  neuron-explorer view -n "$neff" -s "$f" --output-format summary-json > "${f%.ntff}.summary.json" 2> "${f%.ntff}.view.log"
done
ls -la "$out"
exit $rc
