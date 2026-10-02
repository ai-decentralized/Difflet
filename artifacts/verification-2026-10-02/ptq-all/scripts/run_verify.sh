#!/usr/bin/env bash
# run_verify.sh <bf16-slug> : run scripts/ptq_model_verify.sh in the worktree venv (DRY=1 honoured).
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
bash scripts/ptq_model_verify.sh "$@"
