#!/usr/bin/env bash
# chain_capture.sh <key> <arm> <device> <cpus>: once the key's warm lane is done, its capture lane
K=$1 A=$2 D=$3 C=$4
cd /mnt/nvme/difflet-mxattn/task6/logs || exit 1
until grep -q WARM_LANE_DONE "warm_$K.out"; do sleep 15; done
exec ./capture_lane.sh "$K" "$A" "$D" "$C" > "capture_$K.out" 2>&1
