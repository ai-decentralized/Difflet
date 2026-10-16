#!/bin/bash
# usage: detach.sh <log> <script> [args...]  -> prints the detached PID
log="$1"; shift
setsid nohup bash "$@" > "$log" 2>&1 < /dev/null &
echo $!
