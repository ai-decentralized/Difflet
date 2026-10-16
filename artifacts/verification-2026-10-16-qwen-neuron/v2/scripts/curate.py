"""Curate V2 evidence (CP on device) into the worktree's artifacts dir: logs, per-rank JSON,
summary.json and a table; no tensors. Extra run dirs (e.g. a post-probe health rerun) as argv."""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import torch

WORK = Path("/home/ubuntu/qwen-v2-work")
DEST = Path("/home/ubuntu/Difflet/.claude/worktrees/qwen-image-neuron/artifacts/"
            "verification-2026-10-16-qwen-neuron/v2")
RUNS = {
    "l4-tp2cp2-gather_kv.try1-nohook": "4 layers tp2cp2 gather_kv, cold first launch (before "
                                       "the cp_scatter check was added)",
    "l4-tp2cp2-gather_kv": "4 layers tp2cp2 gather_kv, warm relaunch with the cp_scatter check",
    "l4-tp2cp2-ulysses": "4 layers tp2cp2 ulysses (cold block NEFF)",
    "l4-tp2cp2-ring": "4 layers tp2cp2 ring (cold block NEFF)",
    "l4-tp1cp4-ulysses": "4 layers tp1cp4 ulysses (cold block NEFF)",
    "l4-tp1cp4-ring": "4 layers tp1cp4 ring (cold block NEFF)",
    "l4-tp1cp4-gather_kv": "4 layers tp1cp4 gather_kv (cold block NEFF)",
    "l60-tp2cp2-ulysses": "60 layers tp2cp2 ulysses (block NEFF warm from the 4-layer run)",
    "l60-tp2cp2-gather_kv": "60 layers tp2cp2 gather_kv, vs V1 TP4 compile and vs ulysses",
    "l60-tp2cp2-ring": "60 layers tp2cp2 ring, vs V1 TP4 compile and vs ulysses",
    "l60-tp2cp2-ulysses-warm": "60 layers tp2cp2 ulysses relaunch: --expect warm, --equal-to "
                               "the first launch, vs V1 TP4",
    "l30-tp1cp4-ulysses": "30 layers tp1cp4 ulysses (a larger N that fits tp1)",
    "l30-tp1cp4-gather_kv": "30 layers tp1cp4 gather_kv, vs ulysses",
    "l30-tp1cp4-ring": "30 layers tp1cp4 ring, vs ulysses",
}
for extra in sys.argv[1:]:
    RUNS[extra] = extra
DEST.mkdir(parents=True, exist_ok=True)
(DEST / "scripts").mkdir(exist_ok=True)
for name in ("dev.sh", "detach.sh", "cell.sh", "wait.sh", "curate.py"):
    shutil.copy2(WORK / name, DEST / "scripts" / name)

ELIDE = re.compile(r"(were not used when initializing QwenImageTransformer2DModel:\s*)\[.*?\]", re.S)
NOISE = re.compile(r"nrt_infodump|TDRV:|NRT:|^\s*\\_|^ND |^\s+\||^\s*$|NEFF id|^\d{4}: |^\(\*\)"
                   r"|^\s*\|")
for log in sorted(WORK.glob("*.log")):
    text = ELIDE.sub(r"\1[... unused block keys elided ...]", log.read_text(errors="replace"))
    if "probe" in log.name:
        # The first runtime OOM report in full, then the Python-level lines and a count.
        lines = text.splitlines()
        first = []
        for line in lines:
            first.append(line)
            if "cut to here" in line:
                break
        rest = [ln for ln in lines[len(first):] if not NOISE.search(ln)]
        oom = [ln for ln in rest if "Neuron OOM" in ln or "OOM recovery" in ln]
        other = [ln for ln in rest if "Neuron OOM" not in ln and "OOM recovery" not in ln]
        text = "\n".join(
            first
            + [f"[curated: {len(lines)} lines in the raw log, {len(oom)} 'Neuron OOM' / 'OOM "
               "recovery' warnings; the first 12 and last 12 follow, then every line that is "
               "not part of a runtime info dump]"]
            + oom[:12] + ["..."] + oom[-12:] + other) + "\n"
    (DEST / log.name).write_text(text)

summary = {"host": "trn2.3xlarge, 4 logical NeuronCores (config 2)",
           "cache_root": "/home/ubuntu/.cache/difflet-neuron-qwen (shared with V1)",
           "shape": "1024x1024 (image_seq_len 4096), text_seq_len 1024, bf16, compile mode",
           "runs": {}}
ref = torch.load(WORK / "ref-l30.pt", weights_only=False)
summary["references"] = {
    "ref-l30.pt (new in V2)": {
        "num_layers": ref["num_layers"], "prompt": ref["prompt"], "valid_rows": ref["valid_rows"],
        "timesteps": ref["timesteps"], "timings_s": ref["timings"],
        "loading_info_counts": {k: len(v) for k, v in ref["loading_info"].items()},
        "fp32_output_sha256": {f"{r['timestep']:g}": hashlib.sha256(
            r["reference"].contiguous().numpy().tobytes()).hexdigest() for r in ref["runs"]}},
    "ref-l4.pt, ref-l60.pt": "V1's (see v1/summary.json references)"}


def run_summary(run, label):
    src = WORK / run
    (DEST / run).mkdir(exist_ok=True)
    ranks = []
    for f in sorted(src.glob("result*.json")):
        shutil.copy2(f, DEST / run / f.name)
        ranks.append(json.loads(f.read_text()))
    ranks.sort(key=lambda r: r["rank"])
    r0 = ranks[0]
    bench = [r["benchmark"] for r in ranks if r["benchmark"]]
    par = {}
    for ts, p in r0["parity"].items():
        e = {k: p[k] for k in ("ok", "ratio_vs_control", "ratio_l2_vs_control") if k in p}
        if "vs_fp32" in p:
            e |= {m: p["vs_fp32"][m] for m in ("cosine", "rel_l1", "rel_l2", "rel_max", "err_cos")}
            e |= {f"control_{m}": p["control"][m] for m in ("cosine", "rel_l1", "rel_l2")}
        e |= {k: {m: p[k][m] for m in ("cosine", "rel_l1", "rel_l2", "rel_max")}
              for k in p if k.startswith("vs_") and k != "vs_fp32"}
        par[ts] = e
    checks = {c["name"]: c for c in r0["checks"]}
    return {
        "label": label,
        "cfg": {k: r0["config"][k] for k in ("num_layers", "tp", "cp", "cp_mode",
                                             "heads_per_rank")},
        "shape_rules": r0.get("shape_rules"),
        "ok_all_ranks": all(r["ok"] for r in ranks), "n_ranks": len(ranks),
        "checks_rank0": {n: c["ok"] for n, c in checks.items()},
        "failed_checks_any_rank": sorted({f"rank{r['rank']}:{c['name']}" for r in ranks
                                          for c in r["checks"] if not c["ok"]}),
        "ranks_identical_detail": checks.get("ranks_identical", {}).get("detail"),
        "cp_scatter_detail": checks.get("cp_scatter", {}).get("detail"),
        "parity": par,
        "phase_seconds_rank_max": {k: round(max(r["phase_seconds"].get(k, 0) for r in ranks), 2)
                                   for k in r0["phase_seconds"]},
        "prepare_runtime_s_rank_max": round(max(r["timers_s"]["prepare_runtime"]
                                                for r in ranks), 2),
        "adapter_forward_ms_per_rank": [[round(s * 1e3, 1) for s in
                                         r["timers_s"]["adapter_forwards"]] for r in ranks],
        "bench_median_ms_per_rank": [round(b["median_ms"], 2) for b in bench],
        "bench_p10_p90_ms_rank0": [round(bench[0]["p10_ms"], 2), round(bench[0]["p90_ms"], 2)]
        if bench else None,
        "bench_samples_per_rank": len(bench[0]["samples_ms"]) if bench else 0,
        "peak_hbm_gib_per_rank": [round(r["max_memory_allocated_gib"], 3) for r in ranks],
        "weights_gib_per_rank": [round(r["weights_allocated_gib"], 3) for r in ranks],
        "host_vmhwm_gib_per_rank": [round(r["host_vmhwm_gib_by_phase"]["end"], 2) for r in ranks],
        "run_counters_per_rank": [r["counters"]["run"] for r in ranks],
        "warmup_counters_per_rank": [r["counters"]["warmup"] for r in ranks],
        "forward_and_bench_counters_per_rank": [r["counters"]["forwards_and_benchmark"]
                                                for r in ranks],
        "new_neffs_in_cache": r0["new_neffs"],
        "block_neffs_per_rank": [[(b["key"][:12], b["neff_size_bytes"],
                                   round(b["compilation_time_ms"] / 1e3, 1))
                                  for b in r["block_neffs"]] for r in ranks],
        "compiled_blocks": r0["compiled_blocks"],
        "fallbacks_all_ranks": [r["device_evidence"]["fallbacks"] + r["warmup_fallbacks"]
                                + r["benchmark_fallbacks"] for r in ranks],
        "output_sha256_rank0": {ts: o["sha256"] for ts, o in r0["outputs"].items()},
        "load_report": r0["load_report"],
    }


for run, label in RUNS.items():
    summary["runs"][run] = run_summary(run, label)
(DEST / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

rows = ["| cell | ok, ranks | cos vs fp32 t1000/t500 | rel_l1 / control t1000/t500 | "
        "fwd median ms (rank 0, n) | peak HBM GiB | weights GiB | warm-up s (rank max) | "
        "block NEFF keys | compile s by rank |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for run, s in summary["runs"].items():
    p = s["parity"]
    cos = "/".join(f"{p[t].get('cosine', float('nan')):.6f}" for t in p)
    rat = "/".join(f"{p[t].get('ratio_vs_control', float('nan')):.3f}" for t in p)
    keys = sorted({b[0] for r in s["block_neffs_per_rank"] for b in r})
    comp = [b[2] for r in s["block_neffs_per_rank"] for b in r]
    med = s["bench_median_ms_per_rank"][0] if s["bench_median_ms_per_rank"] else "-"
    rows.append(f"| {run} | {s['ok_all_ranks']}, {s['n_ranks']} | {cos} | {rat} | {med} "
                f"({s['bench_samples_per_rank']}) | {max(s['peak_hbm_gib_per_rank']):.2f} | "
                f"{max(s['weights_gib_per_rank']):.2f} | "
                f"{s['phase_seconds_rank_max'].get('warmup forward')} | {', '.join(keys)} | "
                f"{comp} |")
(DEST / "table.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
print("wrote", DEST)
