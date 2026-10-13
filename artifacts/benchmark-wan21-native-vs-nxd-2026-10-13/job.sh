#!/bin/bash
# Wan2.1 TorchNeuron vs NxD AoT benchmark, VCS=2. Resumable: recorded runs are skipped.
W=/home/ubuntu/Difflet/.claude/worktrees/wan21-bench
OUT=/home/ubuntu/wan21-bench
cd $OUT
if python3 $W/benchmark/wan21_native_vs_nxd.py run --backend neuron --out $OUT --python /home/ubuntu/workspace/native_venv/bin/python \
 && python3 $W/benchmark/wan21_native_vs_nxd.py run --backend trainium --out $OUT --python /home/ubuntu/Difflet/.venv/bin/python \
 && python3 $W/benchmark/wan21_native_vs_nxd.py report --out $OUT; then
  echo "JOB OK" > $OUT/DONE
else
  echo "JOB FAILED rc=$?" > $OUT/DONE
fi
