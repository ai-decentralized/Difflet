"""Curate V1 evidence into the worktree's artifacts dir (logs, per-rank JSON, summary; no tensors)."""
import hashlib
import json
import re
import shutil
import statistics
from pathlib import Path

import torch

WORK = Path("/home/ubuntu/qwen-v1-work")
DEST = Path("/home/ubuntu/Difflet/.claude/worktrees/qwen-image-neuron/artifacts/"
            "verification-2026-10-16-qwen-neuron/v1")
RUNS = {
    "a-l4-eager": "(a) 4 layers TP4 eager vs CPU fp32 (scratch cache root difflet-neuron-qwen-l4)",
    "b-l4-compile-cold": "(b) 4 layers TP4 compile, cold block NEFF, vs CPU fp32 and vs (a)",
    "c-l60-compile-cold": "(c) 60 layers TP4 compile, empty cache root difflet-neuron-qwen",
    "c-l60-compile-warm": "(c) 60 layers TP4 compile, second launch on the warm cache",
    "d-l60-eager": "(d) 60 layers TP4 eager vs CPU fp32 and vs (c) cold",
}
DEST.mkdir(parents=True, exist_ok=True)
(DEST / "scripts").mkdir(exist_ok=True)
for name in ("dev.sh", "ref.sh", "detach.sh", "summ.py", "curate.py"):
    shutil.copy2(WORK / name, DEST / "scripts" / name)

# Logs: the reference log without diffusers' list of the 1792 unused block tensors.
for log in sorted(WORK.glob("*.log")):
    text = log.read_text(errors="replace")
    text = re.sub(r"(were not used when initializing QwenImageTransformer2DModel: )\[.*?\]",
                  r"\1[... 1792 keys of transformer_blocks.4-59 elided ...]", text, flags=re.S)
    (DEST / log.name).write_text(text)

summary = {"host": "trn2.3xlarge, 4 logical NeuronCores (config 2)", "runs": {}}
for ref_name in ("ref-l4.pt", "ref-l60.pt"):
    ref = torch.load(WORK / ref_name, weights_only=False)
    summary.setdefault("references", {})[ref_name] = {
        "num_layers": ref["num_layers"], "prompt": ref["prompt"], "valid_rows": ref["valid_rows"],
        "timesteps": ref["timesteps"],
        "dit_timesteps": [float(r["dit_timestep"][0]) for r in ref["runs"]],
        "timings_s": ref["timings"], "threads": ref["threads"], "versions": ref["versions"],
        "loading_info_counts": {k: len(v) for k, v in ref["loading_info"].items()},
        "fp32_output_sha256": {f"{r['timestep']:g}": hashlib.sha256(
            r["reference"].contiguous().numpy().tobytes()).hexdigest() for r in ref["runs"]},
    }
emb = torch.load(DEST.parent / "prompt_embeds.pt", weights_only=False)
summary["prompt_embeds"] = {k: emb[k] for k in ("prompt", "valid_rows", "encoder_dtype",
                                                 "load_s", "encode_s", "stats")}
summary["prompt_embeds"]["sha256"] = hashlib.sha256(
    emb["embeds"].view(torch.int16).contiguous().numpy().tobytes()).hexdigest()

for run, label in RUNS.items():
    src = WORK / run
    (DEST / run).mkdir(exist_ok=True)
    ranks = []
    for f in sorted(src.glob("result*.json")):
        shutil.copy2(f, DEST / run / f.name)
        ranks.append(json.loads(f.read_text()))
    ranks.sort(key=lambda r: r["rank"])
    r0 = ranks[0]
    bench = [r["benchmark"] for r in ranks if r["benchmark"]]
    summary["runs"][run] = {
        "label": label,
        "ok_all_ranks": all(r["ok"] for r in ranks),
        "checks_rank0": {c["name"]: c["ok"] for c in r0["checks"]},
        "parity": {ts: {k: v for k, v in p.items() if k in (
            "ok", "ratio_vs_control", "ratio_l2_vs_control", "gated", "note")}
                    | ({"cosine": p["vs_fp32"]["cosine"], "rel_l1": p["vs_fp32"]["rel_l1"],
                        "rel_l2": p["vs_fp32"]["rel_l2"], "rel_max": p["vs_fp32"]["rel_max"],
                        "err_cos": p["vs_fp32"]["err_cos"],
                        "control_cosine": p["control"]["cosine"],
                        "control_rel_l1": p["control"]["rel_l1"],
                        "control_rel_l2": p["control"]["rel_l2"]} if "vs_fp32" in p else {})
                    | {k: {m: p[k][m] for m in ("cosine", "rel_l1", "rel_l2", "rel_max")}
                       for k in p if k.startswith("vs_") and k != "vs_fp32"}
                   for ts, p in r0["parity"].items()},
        "phase_seconds_rank_max": {k: max(r["phase_seconds"].get(k, 0) for r in ranks)
                                   for k in r0["phase_seconds"]},
        "prepare_runtime_s_rank_max": max(r["timers_s"]["prepare_runtime"] for r in ranks),
        "adapter_forward_ms_per_rank": [[round(s * 1e3, 1) for s in r["timers_s"]["adapter_forwards"]]
                                        for r in ranks],
        "bench_median_ms_per_rank": [round(b["median_ms"], 2) for b in bench],
        "bench_p10_p90_ms_rank0": [round(bench[0]["p10_ms"], 2), round(bench[0]["p90_ms"], 2)]
        if bench else None,
        "bench_samples_per_rank": len(bench[0]["samples_ms"]) if bench else 0,
        "bench_spread_drift_rank0": [round(bench[0]["spread"], 4), round(bench[0]["drift"], 4)]
        if bench else None,
        "peak_hbm_gib_per_rank": [round(r["max_memory_allocated_gib"], 3) for r in ranks],
        "weights_gib_per_rank": [round(r["weights_allocated_gib"], 3) for r in ranks],
        "host_vmhwm_gib_per_rank": [round(r["host_vmhwm_gib_by_phase"]["end"], 2) for r in ranks],
        "run_counters_per_rank": [r["counters"]["run"] for r in ranks],
        "warmup_counters_per_rank": [r["counters"]["warmup"] for r in ranks],
        "forward_and_bench_counters_per_rank": [r["counters"]["forwards_and_benchmark"]
                                                for r in ranks],
        "compilation_time_s_per_rank": [round(r["snapshots"]["end"]["compilation_time"]["total_s"], 1)
                                        for r in ranks],
        "new_neffs": r0["new_neffs"],
        "block_neffs_per_rank": [[(b["key"][:12], b["neff_size_bytes"],
                                   round(b["compilation_time_ms"] / 1e3, 1))
                                  for b in r["block_neffs"]] for r in ranks],
        "compiled_blocks": r0["compiled_blocks"],
        "fallbacks_all_ranks": [r["device_evidence"]["fallbacks"] + r["warmup_fallbacks"]
                                + r["benchmark_fallbacks"] for r in ranks],
        "unwarmed_shapes_all_ranks": [r["device_evidence"]["unwarmed_shapes"] for r in ranks],
        "output_stats_rank0": r0["output_stats"],
        "output_sha256_rank0": {ts: o["sha256"] for ts, o in r0["outputs"].items()},
        "load_report": r0["load_report"],
        "pg_timeout": r0["pg_timeout"],
        "torch_threads": r0["torch_threads"],
    }
cold = summary["runs"]["c-l60-compile-cold"]["phase_seconds_rank_max"]["warmup forward"]
warm = summary["runs"]["c-l60-compile-warm"]["phase_seconds_rank_max"]["warmup forward"]
summary["derived"] = {
    "l60_cold_compile_wall_s": {"value": cold - warm, "definition":
                                "rank-max 'warmup forward' cold launch minus warm launch",
                                "cold_warmup_s": cold, "warm_warmup_s": warm},
    "l60_compile_vs_eager_speedup": statistics.median(
        summary["runs"]["d-l60-eager"]["bench_median_ms_per_rank"]) / statistics.median(
        summary["runs"]["c-l60-compile-warm"]["bench_median_ms_per_rank"]),
}
(DEST / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["derived"], indent=2))
print("wrote", DEST)
