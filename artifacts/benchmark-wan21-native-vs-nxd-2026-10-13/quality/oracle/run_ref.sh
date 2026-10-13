#!/bin/bash
set -euo pipefail
unset DIFFLET_BACKEND NEURON_RT_NUM_CORES NEURON_LOGICAL_NC_CONFIG
export HF_HUB_OFFLINE=1 PYTHONSAFEPATH=1 PYTHONPATH=/home/ubuntu/Difflet/.claude/worktrees/wan21-bench
cd /home/ubuntu/wan21-bench/quality/oracle
echo "start $(date -Is)"
/usr/bin/time -v /home/ubuntu/workspace/native_venv/bin/python /home/ubuntu/Difflet/.claude/worktrees/wan21-bench/tests/numerical/wan_pipeline_reference.py \
  --model-dir /home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --prompt "a cinematic shot of a red fox running through a snowy forest" --steps 2 --guidance-scale 5.0 --seed 42 --threads 12 \
  --out /home/ubuntu/wan21-bench/quality/oracle/ref_2steps.pt
echo "end $(date -Is) rc=$?"
