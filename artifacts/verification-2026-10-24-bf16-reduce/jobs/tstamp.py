"""Prefix each stdin line with seconds since this process started (phase breakdown)."""
import sys
import time

start = time.time()
for line in sys.stdin:
    sys.stdout.write(f"[{time.time() - start:9.3f}] {line}")
    sys.stdout.flush()
