#!/bin/bash
# Qwen-Image native text-mask fix (2026-10-28): device before/after runs (not in repo).
#   before: HEAD 549e6f7 (snapshot nf_qwen_text/src_head), root copy of qwen_image_native_tp4_p3
#   after:  the fix (worktree native-followups, uncommitted), its own root copy (recompiles blocks)
# Every run takes /home/ubuntu/neuron-device.lock (shared with the HunyuanVideo agent) and
# releases it after. Flags exactly native_tp4_p3's (--backend neuron --tp-degree 4 --vae-dtype bf16).
#   systemd-run --user --unit=nfq_dev --collect -- bash job_device.sh [runs...]
# Host paths are settings (2026-11-03, moved into the repo): BENCH4M_DIR is the data root (plus
# bench4m.py's own BENCH4M_* settings), NFQ_AFTER_WT the fixed worktree, NFQ_BEFORE_WT the
# unfixed one (default: the snapshot dir it was), NEURON_DEVICE_LOCK the shared lock file.
# bench4m_nfq.py differed from bench4m.py only in honouring BENCH4M_NATIVE_WT, which
# bench4m.py now does, so this calls bench4m.py.
B=${BENCH4M_DIR:?set BENCH4M_DIR (see benchmark/bench4m/README.md)}
S=$B/nf_qwen_text
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=${NEURON_DEVICE_LOCK:-$HOME/neuron-device.lock}
mkdir -p $S
exec >> $S/device.log 2>&1
cd $B
BEFORE_WT=${NFQ_BEFORE_WT:-$S/src_head}
AFTER_WT=${NFQ_AFTER_WT:?set NFQ_AFTER_WT to the worktree with the fix}
run() {  # cell run wt [bench4m args...]
  local cell=$1 r=$2 wt=$3; shift 3
  if [ -f $B/qwen_image/$cell/$r/summary.json ]; then echo "[job] $cell/$r exists, skip"; return 0; fi
  echo "[job] $cell/$r start $(date -u +%FT%TZ) wt=$wt"
  BENCH4M_NATIVE_WT=$wt flock $LOCK /usr/bin/python3 $HERE/bench4m.py \
    run --model qwen_image --cell $cell --run $r --mode generate --venv native "$@" \
    -- --backend neuron --tp-degree 4 --vae-dtype bf16
  local rc=$?
  echo "[job] $cell/$r rc=$rc $(date -u +%FT%TZ)"; echo "$cell/$r rc=$rc $(date -u +%FT%TZ)" >> $S/device.rcs
  return $rc
}
echo "[job] start $(date -u +%FT%TZ) pid $$ runs: $*"
for spec in "$@"; do
  cell=${spec%/*}; r=${spec#*/}
  case $cell in nfq_before) wt=$BEFORE_WT;; nfq_after) wt=$AFTER_WT;; *) echo "bad $spec"; exit 2;; esac
  extra=()
  [[ $r == q* ]] && extra=(--env HV1_DUMP_LATENTS_DIR=$B/qwen_image/$cell/$r)
  run $cell $r $wt "${extra[@]}" || { echo "FAIL $spec" > $S/device.done; exit 1; }
done
echo "[job] ALL_DONE $(date -u +%FT%TZ)"; echo "ALL_DONE $* $(date -u +%FT%TZ)" > $S/device.done
