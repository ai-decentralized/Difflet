#!/bin/bash
set -e
W=/home/ubuntu/qwen-v3-work
V=/home/ubuntu/Difflet/.claude/worktrees/qwen-image-neuron/artifacts/verification-2026-10-16-qwen-neuron/v3
mkdir -p $V/diag $V/scripts/probe $V/png
cp $W/*.log $V/
cp $W/diag/*.log $V/diag/
for r in tp4-compile tp4-s50-gen1 tp4-s50-gen2 tp4-s50-gen3-noprobe tp4-s30 tp2cp2-ulysses-s50 tp4-eager-s50 cold-compile cold-gen; do cp $W/res-$r.json $V/$r.result.json; done
cp $W/diag/res-*.json $W/diag/traj_compare.json $V/diag/
cp $W/compare.json $W/embeds_parity.json $W/summary.json $V/
cp $W/tp4-s50-gen1/out.png $V/png/tp4-s50-seed42.png
cp $W/tp4-s30/out.png $V/png/tp4-s30-seed42.png
cp $W/tp2cp2-ulysses-s50/out.png $V/png/tp2cp2-ulysses-s50-seed42.png
cp $W/tp4-eager-s50/out.png $V/png/tp4-eager-s50-seed42.png
cp $W/cli.sh $W/cli-cold.sh $W/gen.sh $W/gen-eager.sh $W/gen-cold.sh $W/diag.sh $W/diag-batch.sh $W/cold-batch.sh $W/embeds.sh $W/detach.sh $W/summ.py $W/cmp.py $W/traj_cmp.py $W/derive.py $W/curate.sh $V/scripts/
cp $W/probe/sitecustomize.py $V/scripts/probe/
du -sh $V
find $V -type f | wc -l
