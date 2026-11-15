#!/usr/bin/env bash
# Task 6 step 4: Plan A's block capture of a key's quantized arm (capture.sh, its dump on the
# NVMe cache mirror), analyze.py on each kept block, the breakdowns and lane files copied beside
# Plan A's committed bf16 assets.
# usage: capture_lane.sh <key> <arm> <device> <cpus>
set -u
K=$1 A=$2 D=$3 C=$4
WT=/home/ubuntu/Difflet/.claude/worktrees/mxfp8-sage-attention
T=/mnt/nvme/difflet-mxattn/task6
export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
# scratch off the root disk: NKI traces (tempfile) and neuronx-cc work dirs
export TMPDIR=/mnt/nvme/difflet-mxattn/task6/tmp TORCH_NEURONX_DEBUG_DIR=/mnt/nvme/difflet-mxattn/task6/ccwork
export PY=$HOME/venvs/native-fp8/bin/python
export CAPTURE_LOCK=$T/.capture.lock DUMP_SLOT=$T/.dump-slot
case "$K" in qwen_image) export CAPTURE_CACHE=$T/cache/fp8 ;; *) export CAPTURE_CACHE=$T/cache/video ;; esac
case "$K $A" in  # run_all.sh's hub_of, mxfp8's for both, bf16's for attn
  "qwen_image both"|ltx_2\ *|wan21_official\ *|wan22_official\ *) export HF_HUB_CACHE=$HOME/.cache/huggingface/hub ;;
  *) export HF_HUB_CACHE=/mnt/nvme/difflet-quant-bench-hub ;;
esac
echo "$(date '+%F %T') capture $K $A dev $D cpus $C hub $HF_HUB_CACHE cache $CAPTURE_CACHE df: $(df --output=avail -BG / /mnt/nvme | tail -2 | tr -s ' \n' ' ')"
taskset -c "$C" "$WT/scripts/profile_dit/capture.sh" --key "$K" --arm "$A" --device "$D" --cpus "$C" --out "$T/prof"
echo "$(date '+%F %T') capture.sh exit $? STATUS $(cat $T/prof/$K/$A/STATUS)"
L=$T/prof/$K/$A
[ "$(head -1 $L/STATUS)" = DONE ] || exit 1
DST=$WT/docs/verification/assets/2026-11-13-dit-block-profile/$K/$A
mkdir -p "$DST"
cp "$L/system_summary.json" "$DST/"
for blk in "$L"/blk*/; do
  b=$(basename "$blk")
  taskset -c "$C" "$PY" "$WT/scripts/profile_dit/analyze.py" "$blk" --model "$K" > "$blk/analyze.log" 2>&1
  echo "$(date '+%F %T') analyze $b exit $?: $(tail -1 $blk/analyze.log)"
  mkdir -p "$DST/$b"
  cp "$blk/source.json" "$DST/$b/"
  cp "$blk/breakdown.json" "$DST/$b.json"
done
echo CAPTURE_LANE_DONE
