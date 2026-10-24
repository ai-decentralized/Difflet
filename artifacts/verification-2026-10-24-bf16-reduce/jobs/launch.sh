#!/usr/bin/env bash
# launch.sh <name> <job-script> [args...]: run a job fully detached (own session,
# no controlling terminal) so it outlives any monitor or tool call. Writes
# $CAMPAIGN/jobs/<name>.{pid,log} and appends "<name> exit=<rc>" to
# $CAMPAIGN/jobs/exits.txt when it ends.
CAMPAIGN=/home/ubuntu/campaign/bf16-reduce
name=$1
shift
setsid nohup bash -c '
  name=$1; shift
  bash "$@"
  echo "$name exit=$?" >> /home/ubuntu/campaign/bf16-reduce/jobs/exits.txt
' _ "$name" "$@" > "$CAMPAIGN/jobs/$name.log" 2>&1 < /dev/null &
echo $! > "$CAMPAIGN/jobs/$name.pid"
echo "launched $name pid $(cat "$CAMPAIGN/jobs/$name.pid")"
