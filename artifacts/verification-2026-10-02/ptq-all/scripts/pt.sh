#!/usr/bin/env bash
# pt.sh <pytest args...> : run pytest in the worktree venv, filtered output.
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
pytest -q -p no:cacheprovider "$@" 2>&1 | grep -vE 'Warning|warnings.warn|^\s*$' | tail -${PT_TAIL:-20}
