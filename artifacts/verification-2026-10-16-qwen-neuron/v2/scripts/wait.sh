#!/bin/bash
# usage: wait.sh <pid> <log>  (waits up to 580 s, then prints the summary lines)
timeout 580 tail --pid=$1 -f /dev/null
if kill -0 $1 2>/dev/null; then echo "STILL RUNNING pid=$1"; tail -3 $2; exit 0; fi
grep -E "^(PASS|FAIL) |\[rank [0-9]\] (PASS|FAIL)|EXIT rc|Error|error:|Traceback" $2 | head -60
