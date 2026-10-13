#!/bin/bash
# fp32-oracle W4 parity of the fixed build's 2-step g=5 latents (host only).
W=/home/ubuntu/Difflet/.claude/worktrees/wan21-bench
mkdir -p /home/ubuntu/wan21-bench/ccop-verify/oracle
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONSAFEPATH=1 PYTHONPATH=$W \
  /home/ubuntu/workspace/native_venv/bin/python $W/benchmark/wan21_oracle_compare.py \
  --ref /home/ubuntu/wan21-bench/quality/oracle/ref_2steps.pt \
  --device neuron=/home/ubuntu/wan21-bench/ccop-verify/twostep-g5/probe/latents-134982-1.pt \
  --out /home/ubuntu/wan21-bench/ccop-verify/oracle/compare.json
echo "rc=$?"
