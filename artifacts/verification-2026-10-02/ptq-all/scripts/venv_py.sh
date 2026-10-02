#!/usr/bin/env bash
# venv_py.sh <script.py> [args...] : run a python script in the worktree venv with PYTHONPATH set.
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan || exit 1
source .venv/bin/activate
export PYTHONPATH=$PWD
exec python "$@"
