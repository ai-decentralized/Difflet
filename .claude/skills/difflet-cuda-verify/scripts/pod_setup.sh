#!/usr/bin/env bash
# pod_setup.sh: host setup ON a Yottalabs pod (the F1b / F2 recipe), after ship_branch.sh.
# Usage:
#   cd ~/Difflet && bash .claude/skills/difflet-cuda-verify/scripts/pod_setup.sh [--volume]
#   --volume   the difflet-cache network volume is mounted at /data (us-east-3 only):
#              HF_HOME=/data/hf, and /data is chowned to the login user once (when not theirs).
#   (default)  no volume: HF_HOME=$HOME/hf on the container disk (mind containerVolumeInGb:
#              a disk overrun restarted the container in an earlier session).
# Steps: uv (installed if missing), `uv venv --python 3.11 .venv` (kept when present), torch
# 2.11.0 from the cu128 index, `-e ".[test,cuda]"`, `huggingface_hub[cli]`, `imageio[ffmpeg]`,
# then scripts/cuda_env_check.sh into ~/logs/env-check-<ts>.log. HF_HOME is appended to
# ~/.bashrc once. The venv has no pip: install with `uv pip install --python .venv/bin/python`.
# Idempotent: re-running reinstalls nothing that is already satisfied.
set -euo pipefail
case "${1:-}" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p;}' "$0"; exit 0 ;;
  --volume) VOLUME=1 ;;
  "") VOLUME=0 ;;
  *) echo "pod_setup.sh: unknown argument $1 (see --help)" >&2; exit 2 ;;
esac
REPO=${REPO:-$PWD}
[ -f "$REPO/pyproject.toml" ] && [ -d "$REPO/difflet" ] || {
  echo "pod_setup.sh: $REPO is not a Difflet checkout (run ship_branch.sh first, then cd ~/Difflet)" >&2
  exit 2
}
set -x
if [ "$VOLUME" = 1 ]; then
  mountpoint -q /data || { echo "pod_setup.sh: --volume but nothing is mounted at /data" >&2; exit 2; }
  [ "$(stat -c %U /data)" = "$(id -un)" ] || sudo chown "$(id -un):$(id -gn)" /data
  HF_HOME=/data/hf
else
  HF_HOME=$HOME/hf
fi
export HF_HOME
mkdir -p "$HF_HOME" "$HOME/logs" /tmp/logs
grep -q "HF_HOME=$HF_HOME" ~/.bashrc 2>/dev/null \
  || echo "export HF_HOME=$HF_HOME HF_XET_HIGH_PERFORMANCE=1" >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"
git log --oneline -1
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
uv pip install --python .venv/bin/python -e ".[test,cuda]"
uv pip install --python .venv/bin/python "huggingface_hub[cli]" "imageio[ffmpeg]"
scripts/cuda_env_check.sh 2>&1 | tee "$HOME/logs/env-check-$(date +%Y%m%d-%H%M%S).log"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
df -h "$HOME" "$HF_HOME" /tmp | tail -n +2
