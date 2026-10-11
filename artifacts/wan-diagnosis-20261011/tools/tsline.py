#!/usr/bin/env python3
"""Prefix each stdin line with UTC wall time and seconds since start."""
import sys, time, datetime
t0 = time.monotonic()
for line in sys.stdin:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    sys.stdout.write(f"{now} +{time.monotonic()-t0:8.2f}s | {line}")
    sys.stdout.flush()
