#!/usr/bin/env bash
# Block until the campaign logs gain a line, then print the recent ones.
C=/home/ubuntu/campaign/bf16-reduce
files=("$C/timings.txt" "$C/jobs/exits.txt" "$C/jobs/queue2.log" "$C/jobs/queue3.log")
count() { cat "${files[@]}" 2>/dev/null | wc -l; }
s0=$(count)
until [ "$(count)" != "$s0" ]; do sleep 30; done
sleep 5
tail -n 3 "$C/timings.txt"
tail -n 2 "$C/jobs/queue2.log" "$C/jobs/queue3.log" 2>/dev/null
