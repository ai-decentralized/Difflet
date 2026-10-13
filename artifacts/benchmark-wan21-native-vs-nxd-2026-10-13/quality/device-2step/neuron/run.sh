#!/bin/bash
unset DIFFLET_BACKEND DIFFLET_EXEC_MODE TORCH_NEURONX_NEFF_CACHE_DIR TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR TORCH_NEURONX_HLO_CACHE_DIR NEURON_RT_NUM_CORES NEURON_LOGICAL_NC_CONFIG
export PATH=/home/ubuntu/workspace/native_venv/bin:$PATH
export NEURON_RT_VIRTUAL_CORE_SIZE=2 HF_HUB_OFFLINE=1 DIFFLET_NEURON_PG_TIMEOUT_S=3600
export DIFFLET_COMPILE_CACHE=/home/ubuntu/wan21-bench/neuron/cache
export DIFFLET_WAN_PROBE_DIR=/home/ubuntu/wan21-bench/quality/device-2step/neuron/probe
export PYTHONPATH=/home/ubuntu/Difflet/.claude/worktrees/wan21-bench/benchmark/wan_step_probe:/home/ubuntu/Difflet/.claude/worktrees/wan21-bench
cd /home/ubuntu/wan21-bench/quality/device-2step/neuron
echo "START $(date -u)"
/home/ubuntu/workspace/native_venv/bin/python -m difflet.cli.main generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 --backend neuron --cache-dir /home/ubuntu/wan21-bench/neuron/cache --exec-mode compile --steps 2 --guidance-scale 5.0 --seed 42 --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4
echo "EXIT $? $(date -u)"
