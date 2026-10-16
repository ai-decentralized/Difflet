#!/bin/bash
# usage: cli.sh <run_dir> <probe:0|1> <difflet subcommand + args...>
# Runs the user-facing CLI (python -m difflet.cli.main) once; one device job at a time.
WT=/home/ubuntu/Difflet/.claude/worktrees/qwen-image-neuron
VENV=/home/ubuntu/workspace/native_venv
CACHE=/home/ubuntu/.cache/difflet-neuron-qwen
run="$1"; probe="$2"; shift 2
mkdir -p "$run"
pp=$WT
if [ "$probe" = 1 ]; then pp=$WT:/home/ubuntu/qwen-v3-work/probe; fi
neff_before=$(find $CACHE/neuron/neff -name '*.neff' 2>/dev/null | wc -l)
hlo_before=$(find $CACHE/neuron/hlo -type f 2>/dev/null | wc -l)
echo "launch $(date -u +%FT%TZ) head=$(git -C $WT rev-parse --short HEAD) probe=$probe cache=$CACHE neff_before=$neff_before hlo_files_before=$hlo_before"
echo "cmd: python -m difflet.cli.main $*"
start=$(date +%s.%N)
cd "$run"
env PATH=$VENV/bin:$PATH HF_HUB_OFFLINE=1 DIFFLET_NO_JEMALLOC=1 PYTHONSAFEPATH=1 \
  DIFFLET_NEURON_PG_TIMEOUT_S=7200 DIFFLET_COMPILE_CACHE=$CACHE PYTHONPATH=$pp \
  DIFFLET_V3_PROBE=$([ "$probe" = 1 ] && echo 1) DIFFLET_V3_PROBE_EMBEDS=$([ "$probe" = 1 ] && echo "$run/embeds_rank0.pt") \
  $VENV/bin/python -m difflet.cli.main "$@"
rc=$?
end=$(date +%s.%N)
neff_after=$(find $CACHE/neuron/neff -name '*.neff' 2>/dev/null | wc -l)
hlo_after=$(find $CACHE/neuron/hlo -type f 2>/dev/null | wc -l)
echo "EXIT rc=$rc process_wall_s=$(echo "$end - $start" | bc) start_epoch=$start end_epoch=$end new_neff=$((neff_after-neff_before)) new_hlo_files=$((hlo_after-hlo_before)) $(date -u +%FT%TZ)"
