#!/bin/bash
# Two more warm g=5 launches with the fix, once the page cache has refilled (warm1/warm2 were
# inflated by re-reading weights the NxD runs and the CPU oracle had evicted).
source <(sed -n '/^set -u/,/^}/p' /home/ubuntu/wan21-bench/ccop-verify/run.sh)
run warm4-g5 50 5.0
run warm5-g5 50 5.0
echo done > $B/DONE_EXTRA
