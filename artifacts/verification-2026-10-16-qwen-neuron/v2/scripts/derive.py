"""Add V2's derived numbers to summary.json: speed ratios between CP modes, the tp1cp4 60-layer
HBM probe (from its log), and the shape rules per cell. Every input is a V2 run."""
import json
import re
from pathlib import Path

DEST = Path("/home/ubuntu/Difflet/.claude/worktrees/qwen-image-neuron/artifacts/"
            "verification-2026-10-16-qwen-neuron/v2")
s = json.loads((DEST / "summary.json").read_text())
runs = s["runs"]


def med(run):
    return runs[run]["bench_median_ms_per_rank"][0]


probe_raw = Path("/home/ubuntu/qwen-v2-work/l60-tp1cp4-ulysses-probe.log").read_text(
    errors="replace")
first_oom = re.search(r"^(\S+ \S+) \d+:\d+ ERROR\s+TDRV:dmem_alloc_internal\s+Failed to allocate "
                      r"DEVICE memory \((\d+) bytes\)", probe_raw, re.M)
pools = re.findall(r"allocated=(\d+) reserved=(\d+) active=(\d+) total_hbm=(\d+)", probe_raw)
errors = sorted(set(re.findall(r'error message="([^"]+)"', probe_raw)))
launch = re.search(r"EXIT rc=(\d+) launch_wall_s=([\d.]+)", probe_raw)
GiB = 2 ** 30
s["derived"] = {
    "forward_ms_rank0": {run: med(run) for run in runs},
    "l4_tp2cp2_vs_ulysses": {m: med(f"l4-tp2cp2-{m}") / med("l4-tp2cp2-ulysses")
                             for m in ("gather_kv", "ring")},
    "l4_tp1cp4_vs_ulysses": {m: med(f"l4-tp1cp4-{m}") / med("l4-tp1cp4-ulysses")
                             for m in ("gather_kv", "ring")},
    "l60_tp2cp2_vs_ulysses": {m: med(f"l60-tp2cp2-{m}") / med("l60-tp2cp2-ulysses-warm")
                              for m in ("gather_kv", "ring")},
    "l30_tp1cp4_vs_ulysses": {m: med(f"l30-tp1cp4-{m}") / med("l30-tp1cp4-ulysses")
                              for m in ("gather_kv", "ring")},
    "w1008_vs_w1024_tp2cp2": {"ring": med("l4w1008-tp2cp2-ring") / med("l4-tp2cp2-ring"),
                              "ulysses": med("l4w1008-tp2cp2-ulysses") / med("l4-tp2cp2-ulysses")},
    "shape_rules": {run: r["shape_rules"] for run, r in runs.items()},
    "l60_tp1cp4_hbm_probe": {
        "outcome": "LIMIT: device OOM while loading the weights (tp1 holds the whole model per "
                   "rank)",
        "command": "cell.sh l60-tp1cp4-ulysses-probe 4 ulysses 60 --bench-iters 0",
        "expected_weights_gib_per_rank": 20.430e9 * 2 / GiB,
        "first_runtime_oom": {"time": first_oom.group(1), "request_bytes": int(first_oom.group(2))}
        if first_oom else None,
        "allocator_at_oom_bytes": {"allocated_max": max(int(p[0]) for p in pools),
                                   "total_hbm": int(pools[0][3])} if pools else None,
        "allocator_at_oom_gib": {"allocated_max": max(int(p[0]) for p in pools) / GiB,
                                 "total_hbm": int(pools[0][3]) / GiB} if pools else None,
        "python_errors": errors,
        "raised_at": "difflet/backends/tpu/core/checkpoint.py load_checkpoint_into "
                     "target.copy_ (ranks 2, 3), re-raised by collective_phase sync_status; "
                     "ranks 0, 1 SIGTERM by torchrun",
        "oom_warnings_in_log": probe_raw.count("Neuron OOM: requested_size"),
        "exit_rc": int(launch.group(1)) if launch else None,
        "launch_wall_s": float(launch.group(2)) if launch else None,
        "device_healthy_after": "l4-tp2cp2-ulysses-postprobe: PASS, outputs torch.equal to "
                                "l4-tp2cp2-ulysses, 0 compilations",
    },
}
(DEST / "summary.json").write_text(json.dumps(s, indent=2, default=str))
print(json.dumps({k: v for k, v in s["derived"].items() if k != "shape_rules"}, indent=1))
