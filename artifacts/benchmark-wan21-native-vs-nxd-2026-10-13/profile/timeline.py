"""Print the probe2 event timeline of one run dir, relative to the run's START stamp."""
import glob
import json
import sys

d = sys.argv[1]
only_rank0 = len(sys.argv) > 2 and sys.argv[2] == "rank0"
start = None
end = None
for line in open(d + "/times.txt"):
    k, v = line.split()
    if k == "START":
        start = float(v)
    if k == "END":
        end = float(v)
evs = []
for f in glob.glob(d + "/probe2/events-*.jsonl"):
    for line in open(f):
        evs.append(json.loads(line))
evs.sort(key=lambda e: e["t"])
for e in evs:
    if only_rank0 and e.get("rank") not in (None, "0"):
        continue
    extra = {k: v for k, v in e.items() if k not in ("t", "ev", "pid", "rank")}
    if "exec_start" in extra and extra["exec_start"]:
        extra["exec_start_rel"] = round(extra.pop("exec_start") - start, 2)
    print(f"{e['t'] - start:8.2f}  pid={e['pid']:<6} rank={e.get('rank')!s:<4} {e['ev']:<36} {json.dumps(extra) if extra else ''}")
print(f"{end - start:8.2f}  END")
