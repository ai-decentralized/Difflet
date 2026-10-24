#!/usr/bin/env bash
# Master queue after queue2: rerun queue2 for skipped/failed steps (Flux fp32
# measurement after the d196c82 fix), CPU fp32 references (host only), then
# Task 13 (native Qwen, native Wan2.1 driver, NxD Wan2.1 comparison).
C=/home/ubuntu/campaign/bf16-reduce
while [ ! -e "$C/runs/queue2.done" ]; do sleep 60; done
rm -f "$C/runs/queue2.done"
bash "$C/jobs/queue2.sh"
echo "queue3: queue2 rerun done"
[ -e "$C/runs/cpu_ref/done" ] || bash "$C/jobs/cpu_refs.sh" || echo "queue3: cpu_refs FAILED"
echo "queue3: cpu_refs done"
[ -e "$C/runs/native_qwen/measure.done" ] || bash "$C/jobs/native_qwen.sh" || echo "queue3: native_qwen FAILED"
echo "queue3: native_qwen done"
[ -e "$C/runs/native_wan21/measure.done" ] || bash "$C/jobs/native_wan.sh" || echo "queue3: native_wan FAILED"
echo "queue3: native_wan done"
[ -e "$C/runs/nxd_wan21/measure.done" ] || bash "$C/jobs/nxd_wan21.sh" || echo "queue3: nxd_wan21 FAILED"
touch "$C/runs/queue3.done"
echo "queue3: all done"
