"""Summarise a preflight run dir: per-rank warm-up, counters, block NEFFs, memory, timing."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
files = sorted(d.glob("result*.json"))
rows = []
for f in files:
    r = json.loads(f.read_text())
    b = r.get("benchmark") or {}
    rows.append(r)
    print(f"rank {r['rank']}: ok={r['ok']} warmup={r['phase_seconds'].get('warmup forward', 0):.1f}s "
          f"load_ckpt={r['phase_seconds'].get('load checkpoint', 0):.1f}s "
          f"prepare={r['timers_s']['prepare_runtime']:.1f}s total={r['timers_s']['total']:.1f}s "
          f"run={r['counters']['run']} new_neffs={r['new_neffs']} "
          f"CompilationTime={r['snapshots']['end']['compilation_time']['total_s']:.1f}s "
          f"blocks={[(x['neff_size_bytes'], round(x['compilation_time_ms'] / 1e3, 1)) for x in r['block_neffs']]} "
          f"peak={r['max_memory_allocated_gib']:.2f}GiB weights={r['weights_allocated_gib']:.2f}GiB "
          f"adapter_ms={[round(s * 1e3, 1) for s in r['timers_s']['adapter_forwards']]} "
          f"bench_median={b.get('median_ms', float('nan')):.1f}ms p10={b.get('p10_ms', float('nan')):.1f} "
          f"p90={b.get('p90_ms', float('nan')):.1f} n={len(b.get('samples_ms', []))} "
          f"VmHWM={r['host_vmhwm_gib_by_phase']['end']:.2f}GiB")
    sysfs = r.get("sysfs_device_memory") or {}
    for core, entry in sysfs.items():
        print(f"    sysfs {core}: present={entry.get('present')} peak={entry.get('peak')}")
if rows:
    print("rank-max warmup:", max(r["phase_seconds"].get("warmup forward", 0) for r in rows))
    print("rank-max load checkpoint:", max(r["phase_seconds"].get("load checkpoint", 0) for r in rows))
    print("rank-max peak HBM GiB:", max(r["max_memory_allocated_gib"] for r in rows))
    print("phases rank0:", {k: round(v, 2) for k, v in rows[0]["phase_seconds"].items()})
    print("fallbacks all ranks:", [(r["device_evidence"]["fallbacks"], r["warmup_fallbacks"],
                                    r["benchmark_fallbacks"]) for r in rows])
    print("unwarmed:", [r["device_evidence"]["unwarmed_shapes"] for r in rows])
    print("failed checks:", [(r["rank"], c["name"]) for r in rows for c in r["checks"] if not c["ok"]])
