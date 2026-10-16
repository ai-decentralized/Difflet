#!/bin/bash
# usage: diag.sh <run_dir> <compile|eager> <tp> <cp> <cp_mode> <save_traj.pt> [feed_traj.pt]
# Diagnostic CLI generate (50 steps) recording every DiT call; with a feed, teacher-forced.
run="$1"; em="$2"; tp="$3"; cp="$4"; mode="$5"
export DIFFLET_V3_PROBE_SAVE_TRAJ="$6" DIFFLET_V3_PROBE_STEPS=50
if [ -n "$7" ]; then export DIFFLET_V3_PROBE_FEED="$7"; fi
script=/home/ubuntu/qwen-v3-work/gen.sh
if [ "$em" = eager ]; then script=/home/ubuntu/qwen-v3-work/gen-eager.sh; fi
exec bash $script "$run" 1 50 "$tp" "$cp" "$mode"
