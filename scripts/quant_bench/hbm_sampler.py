"""Read neuron-monitor's JSON stream (``neuron-monitor -c nm.json``) on stdin and, for every runtime
process whose memory sits on one logical core, append one compact line to the output file:

    <unix time> <pid> <core> <device bytes>

so report.py can take each bench step's per-rank device memory peak from the step's time window.
``--cores 0-3,16-19`` keeps only the lanes' cores (neuron-monitor prints about 800 KB per sample
while 64 runtimes are up; the kept lines are what the report reads); without it every core is kept.

usage: neuron-monitor -c nm.json | python hbm_sampler.py OUT [--cores LIST]
"""
import argparse
import json
import sys
import time


def core_set(spec: str) -> set[str]:
    """``"0-3,16"`` -> {"0", "1", "2", "3", "16"}."""
    cores: set[str] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cores.update(str(c) for c in range(int(lo), int(hi or lo) + 1))
    return cores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("--cores", default=None, help="logical cores to keep, e.g. 0-3,16-19")
    args = p.parse_args()
    keep = core_set(args.cores) if args.cores else None
    out = open(args.out, "a", buffering=1)
    for line in sys.stdin:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        now = time.time()
        for rt in d.get("neuron_runtime_data") or []:
            used = (rt.get("report", {}).get("memory_used") or {}).get("neuron_runtime_used_bytes") or {}
            cores = (used.get("usage_breakdown") or {}).get("neuroncore_memory_usage") or {}
            busy = [c for c, b in cores.items() if any(b.values())]
            if len(busy) == 1 and (keep is None or busy[0] in keep):
                out.write(f"{now:.1f} {rt['pid']} {busy[0]} {used.get('neuron_device', 0)}\n")


if __name__ == "__main__":
    main()
