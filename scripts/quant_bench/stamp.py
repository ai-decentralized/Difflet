"""Prefix each line of stdin with the local wall-clock time (``YYYY-MM-DD HH:MM:SS.s``), so a bench
log's stages can be timed afterwards (report.py); the date keeps a run across midnight ordered."""
import sys
import time

for raw in sys.stdin.buffer:
    t = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + ".%d " % int((t % 1) * 10)
    sys.stdout.buffer.write(stamp.encode() + raw)
    sys.stdout.buffer.flush()
