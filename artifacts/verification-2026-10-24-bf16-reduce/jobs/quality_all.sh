#!/usr/bin/env bash
# quality_all.sh <model key> <cpu ref dir name> <device run>: the three comparisons
# (fp32 vs CPU, bf16 vs CPU, bf16 vs fp32) into runs/<model>/quality_*.json.
C=/home/ubuntu/campaign/bf16-reduce/runs
PY=/home/ubuntu/venvs/difflet-neuron/bin/python
Q=/home/ubuntu/Difflet/.claude/worktrees/bf16-tp-reduce-qwen-ltx2/scripts/quality_vs_reference.py
m=$1 ref=$C/cpu_ref/$2 run=${3:-warm1}
"$PY" "$Q" --ref-dir "$ref" --cand-dir "$C/$m/fp32/$run" --out "$C/$m/quality_fp32_vs_cpu.json" 2>&1 | grep -v Warn
"$PY" "$Q" --ref-dir "$ref" --cand-dir "$C/$m/bf16/$run" --out "$C/$m/quality_bf16_vs_cpu.json" 2>&1 | grep -v Warn
"$PY" "$Q" --ref-dir "$C/$m/fp32/$run" --cand-dir "$C/$m/bf16/$run" --out "$C/$m/quality_bf16_vs_fp32.json" 2>&1 | grep -v Warn
