#!/usr/bin/env python3
"""Per-NEFF device time in one DiT step, from rank 0's neuron-explorer system profile JSON
(`view --output-format json` of a `generate` of a step or two under `inspect`): each NEFF's
executions per step and their median `nc_exec_running` time, the other NEFFs' time, the host gaps
between executions and the collectives' own time (`cc_running`), per step. Adapted from
fp8-bench/wan21_tp4/analyze_profile.py, which printed the same over the whole run and counted
every execution twice: the runtime records it once per physical core of the logical core.

The denoise loop is the last burst of block executions (no two consecutive ones more than
STEP_GAP_NS apart), from the start of its first block to the end of its last; per step = its
counts and times over the run's step count.

The dump's own log is the check on both (ruling R9): the steps it says it ran must be the step
count, and the loop window must cover its step intervals (a host gap over STEP_GAP_NS inside the
loop would leave a step outside the window and every per-step number short). HunyuanVideo-1.5's
orchestrator prints "denoise <s> s, <N> steps"; the torchrun and FLUX orchestrators print
"step median <ms> ms (n=<N - 1>, " for two steps or more and nothing at one step (dump_steps
null: unchecked).

usage: system_profile.py <system.json> <rank 0's dump dir> <steps> <dump.log>
                         <block NEFF file name>...
prints JSON, per step: {"blocks", "step": {"wall_ms", "steps", "gap_s", "dump_steps"}, "neffs":
{<NEFF file name as the dump names it>: {"execs_per_step", "in_model_median_ms"}},
"non_block_ms", "host_gap_ms", "cc_running": {"n", "total_ms"}}
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

#: Block executions further apart than this are different phases: the load's warm-up forward,
#: then the prompt encoding, then the denoise loop. Measured: within the loop at most 12.5 ms
#: (Qwen-Image's step boundary; FLUX 0.11, HV1 0.36 ms), from the warm-up to the loop at least
#: 1003 ms (Qwen-Image; FLUX 28.4 s, HV1 23.6 s), 2026-11-08 and 11-13 profiles.
STEP_GAP_NS = 200_000_000


def executions(events: list[dict]) -> list[dict]:
    """One {model_id, start, end} per execution, merged over the physical cores that record it."""
    spans: dict[tuple, list[int]] = {}
    for e in events:
        if e["name"] != "nc_exec_running":
            continue
        if e.get("worker_gid", 0) != 0:
            sys.exit(f"system_profile.py: worker_gid {e['worker_gid']} in the profile: "
                     "not rank 0's")
        start, end = int(e["timestamp"]), int(e["timestamp"]) + int(e["duration"])
        # the run's first execution can come without exec_id (HV1, 2026-11-08); its flow_id is
        # the same on both cores
        key = (e["model_id"], e.get("exec_id", str(e.get("flow_id"))))
        span = spans.setdefault(key, [start, end])
        span[0], span[1] = min(span[0], start), max(span[1], end)
    return sorted(({"model_id": k[0], "start": s, "end": t} for k, (s, t) in spans.items()),
                  key=lambda r: r["start"])


def dump_steps(log: str) -> tuple[int | None, float | None]:
    """(steps, median step interval in ms) as the dump's log prints them; None where it does not."""
    if found := re.findall(r"denoise [\d.]+ s, (\d+) steps", log):
        return int(found[-1]), None
    if found := re.findall(r"step median ([\d.]+) ms \(n=(\d+),", log):
        return int(found[-1][1]) + 1, float(found[-1][0])
    return None, None


def summarize(profile: dict, names: dict[str, str], steps: int, blocks: list[str],
              log: str = "") -> dict:
    runs = executions(profile["trace_event"])
    for r in runs:
        r["neff"] = names.get(str(r["model_id"]), f"model_{r['model_id']}")
    burst = [r for r in runs if r["neff"] in blocks]
    if not burst:
        sys.exit(f"system_profile.py: no execution of {blocks} in the profile")
    first = len(burst) - 1
    while first > 0 and burst[first]["start"] - burst[first - 1]["end"] <= STEP_GAP_NS:
        first -= 1
    t0, t1 = burst[first]["start"], burst[-1]["end"]
    ran, interval_ms = dump_steps(log)
    if ran is not None and ran != steps:
        sys.exit(f"system_profile.py: the dump ran {ran} steps, the lane divides by {steps}")
    # every step interval but the first step's lies between the window's first and last block
    if interval_ms is not None and (t1 - t0) / 1e6 < (ran - 1) * interval_ms:
        sys.exit(f"system_profile.py: the loop window is {(t1 - t0) / 1e6:.1f} ms, the dump's "
                 f"{ran - 1} step interval(s) {(ran - 1) * interval_ms:.1f} ms: a step lies "
                 "outside the loop window")
    # shortcut: the window starts at the loop's first block, so the first forward's embedders
    # (and the last forward's final layer) fall outside it; widen it if the page needs them.
    loop = [r for r in runs if t0 <= r["start"] and r["end"] <= t1]
    by_neff: dict[str, list[int]] = defaultdict(list)
    for r in loop:
        by_neff[r["neff"]].append(r["end"] - r["start"])
    gaps = [b["start"] - a["end"] for a, b in zip(loop, loop[1:])]
    cc = [int(e["duration"]) for e in profile["trace_event"]
          if e["name"] == "cc_running" and t0 <= int(e["timestamp"]) <= t1]
    per_step = steps * 1e6  # ns over the loop -> ms per step
    return {
        "blocks": blocks,
        "step": {"wall_ms": (t1 - t0) / per_step, "steps": steps, "gap_s": STEP_GAP_NS / 1e9,
                 "dump_steps": ran},
        "neffs": {name: {"execs_per_step": len(v) / steps if len(v) % steps else len(v) // steps,
                         "in_model_median_ms": statistics.median(v) / 1e6}
                  for name, v in sorted(by_neff.items(), key=lambda kv: -sum(kv[1]))},
        "non_block_ms": sum(sum(v) for n, v in by_neff.items() if n not in blocks) / per_step,
        "host_gap_ms": sum(g for g in gaps if g > 0) / per_step,
        "cc_running": {"n": len(cc) / steps, "total_ms": sum(cc) / per_step},
    }


def main(path: str, rank0_dir: str, steps: str, dump_log: str, *blocks: str) -> None:
    # the dump names a NEFF neff_<model id>_vnc_<n>.neff; the profile knows its model id
    names = {p.name.split("_")[1]: p.name for p in Path(rank0_dir).rglob("neff_*.neff")}
    print(json.dumps(summarize(json.loads(Path(path).read_text()), names, int(steps),
                               list(blocks), Path(dump_log).read_text(errors="replace")), indent=1))


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit("usage: system_profile.py <system.json> <rank 0's dump dir> <steps> <dump.log> "
                 "<block NEFF>...")
    main(*sys.argv[1:])
