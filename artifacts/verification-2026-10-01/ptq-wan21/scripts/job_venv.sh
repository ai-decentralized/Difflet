#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
PYTHON=python3.12 ./scripts/setup_env.sh
rc=$?
echo "setup_env exit=$rc"
if [ $rc -eq 0 ]; then
  .venv/bin/pip install lpips scikit-image 2>&1 | tail -3
  echo "extras exit=$?"
fi
echo "VENV_DONE rc=$rc"
