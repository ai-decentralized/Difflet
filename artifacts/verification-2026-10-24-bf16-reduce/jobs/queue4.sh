#!/usr/bin/env bash
# After queue3: settle HunyuanVideo's single-run warm e2e (bf16 123.2 s vs fp32
# 109.8 s, mostly load variance). Rank-0 profiles of both DiT NEFFs and two
# extra warm runs per dtype (each after its own page-cache warm-up).
C=/home/ubuntu/campaign/bf16-reduce
N=/var/tmp/neuron-compile-cache/neuronxcc-2.26.6360.0+6f180f47
while [ ! -e "$C/runs/queue3.done" ]; do sleep 60; done
mark() { touch "$C/runs/q4_$1.done"; echo "queue4: $1 done"; }
done_() { [ -e "$C/runs/q4_$1.done" ]; }
done_ profile_hv_fp32 || { bash "$C/jobs/profile.sh" hv_fp32 "$N/MODULE_4f26673a6db893fcb42d+ead70d7c/model.neff"; mark profile_hv_fp32; }
done_ profile_hv_bf16 || { bash "$C/jobs/profile.sh" hv_bf16 "$N/MODULE_6e81cb14fea7eb3470ad+f1d1564e/model.neff"; mark profile_hv_bf16; }
for dt in fp32 bf16; do
  for r in warm2 warm3; do
    done_ "hv_${dt}_$r" || { bash "$C/jobs/measure.sh" hv "$dt" extra "$r" && mark "hv_${dt}_$r"; }
  done
done
touch "$C/runs/queue4.done"
echo "queue4: all done"
