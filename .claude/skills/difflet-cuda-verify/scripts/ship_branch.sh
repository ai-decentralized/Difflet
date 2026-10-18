#!/usr/bin/env bash
# ship_branch.sh: put a local branch on the pod as ~/Difflet, checked out (runs on the laptop).
# Usage:
#   REPO=~/code/Difflet bash .claude/skills/difflet-cuda-verify/scripts/ship_branch.sh <alias> <branch>
#   <alias>   the ~/.ssh/config Host (scripts/yottalabs_pod.py ssh-config ID --alias NAME)
#   <branch>  a local branch; its full history goes over, so the pod needs no GitHub access
# Steps: `git bundle create` of <branch> into a temp file, scp to <alias>:~/difflet.bundle, then
# on the pod: clone it to ~/Difflet the first time, else fetch it and `git checkout -B <branch>`
# (a dirty ~/Difflet stops the checkout; nothing is discarded). Prints the pod's HEAD.
set -euo pipefail
case "${1:-}" in -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p;}' "$0"; exit 0 ;; esac
[ $# -eq 2 ] || { echo "usage: ship_branch.sh <alias> <branch> (see --help)" >&2; exit 2; }
ALIAS=$1; BRANCH=$2
REPO=${REPO:-$PWD}
git -C "$REPO" rev-parse --verify -q "refs/heads/$BRANCH" >/dev/null \
  || { echo "ship_branch.sh: no local branch $BRANCH in $REPO" >&2; exit 2; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
BUNDLE="$TMP/difflet.bundle"
git -C "$REPO" bundle create -q "$BUNDLE" "refs/heads/$BRANCH"
git -C "$REPO" bundle verify -q "$BUNDLE" > /dev/null
echo "[ship] $BRANCH @ $(git -C "$REPO" rev-parse --short "$BRANCH") ($(du -h "$BUNDLE" | cut -f1)) -> $ALIAS"
scp -q "$BUNDLE" "$ALIAS:difflet.bundle"
ssh "$ALIAS" bash -s -- "$BRANCH" <<'POD'
set -euo pipefail
B=$1
if [ ! -d ~/Difflet/.git ]; then
  git clone -q -b "$B" ~/difflet.bundle ~/Difflet
else
  git -C ~/Difflet fetch -q ~/difflet.bundle "+refs/heads/$B:refs/remotes/bundle/$B"
  git -C ~/Difflet checkout -q -B "$B" "refs/remotes/bundle/$B"
fi
echo "[ship] pod HEAD: $(git -C ~/Difflet log --oneline -1)"
POD
