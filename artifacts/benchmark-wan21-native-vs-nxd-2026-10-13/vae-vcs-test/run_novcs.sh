#!/bin/bash
# NxD AoT warm1-g5 repeated with NEURON_RT_VIRTUAL_CORE_SIZE unset (same artifacts),
# to tell whether VCS=2 causes the corrupted on-device VAE decode.
set -u
D=/home/ubuntu/wan21-bench/vae-vcs-test/novcs
mkdir -p $D && cd $D
unset NEURON_RT_VIRTUAL_CORE_SIZE NEURON_RT_NUM_CORES NEURON_LOGICAL_NC_CONFIG DIFFLET_BACKEND
export PATH=/home/ubuntu/Difflet/.venv/bin:$PATH HF_HUB_OFFLINE=1 DIFFLET_NEURON_PG_TIMEOUT_S=3600
export DIFFLET_COMPILE_CACHE=/home/ubuntu/wan21-bench/trainium/cache
export PYTHONPATH=/home/ubuntu/Difflet/.claude/worktrees/wan21-bench/benchmark/wan_step_probe:/home/ubuntu/Difflet/.claude/worktrees/wan21-bench
export DIFFLET_WAN_PROBE_DIR=$D/probe
/home/ubuntu/Difflet/.venv/bin/python -m difflet.cli.main generate \
  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 --backend trainium \
  --cache-dir /home/ubuntu/wan21-bench/trainium/cache \
  --steps 50 --guidance-scale 5.0 --seed 42 \
  --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4 > log.txt 2>&1
echo "rc=$?" > $D/RC
