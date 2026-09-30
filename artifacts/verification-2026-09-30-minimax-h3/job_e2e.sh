#!/usr/bin/env bash
# MiniMax-H3 T2VA e2e at 256x448x124. Runs as a systemd service (MemoryMax guard).
# text artifact already built by `difflet compile` (run 1); remaining stages compiled
# one by one with the stage CLI (same args as the orchestrator), then the production
# `difflet generate` drives all four stages.
set -o pipefail
D=/home/ubuntu/difflet-artifacts/h3-verify-2026-09-30
cd /home/ubuntu/Difflet
export PATH=/home/ubuntu/Difflet/.venv/bin:/opt/aws/neuron/bin:/usr/bin:/bin HOME=/home/ubuntu
SHAPE="--height 256 --width 448 --num-frames 124"
COMMON="--orchestrator MiniMaxAI/MiniMax-H3 --model-id MiniMaxAI/MiniMax-H3 --tp-degree 4 --cp-degree 1 $SHAPE --steps 30 --seed 42 --stage-mode compile --work-dir /home/ubuntu/.cache/difflet/work/minimax-h3"
run() {  # run <name> <cores> <cmd...>
  local name=$1 cores=$2; shift 2
  [ -f $D/$name.done ] && return 0
  local t0=$(date +%s)
  if [ "$cores" = "-" ]; then "$@" > $D/$name.log 2>&1
  else NEURON_RT_NUM_CORES=$cores NEURON_RT_VIRTUAL_CORE_SIZE=2 "$@" > $D/$name.log 2>&1; fi
  local rc=$?
  echo "$name rc=$rc wall=$(( $(date +%s) - t0 ))s" > $D/$name.done
  [ $rc -eq 0 ] || { echo "FAIL $name" > $D/ALL.done; exit 1; }
}
run compile_generate  4 python -m difflet.cli.stage $COMMON --stage generate
run compile_video_vae 1 python -m difflet.cli.stage $COMMON --stage video_vae
run compile_audio_vae 1 python -m difflet.cli.stage $COMMON --stage audio_vae
run e2e_generate - difflet generate --model-id MiniMaxAI/MiniMax-H3 --tp-degree 4 $SHAPE --seed 42 \
  --prompt "A red fox trots through a snowy forest at dawn, its breath visible in the cold air, soft crunching footsteps in the snow" \
  --output $D/fox_256x448x124.mp4 --work-dir $D/work --keep-work-dir
echo "PASS" > $D/ALL.done
