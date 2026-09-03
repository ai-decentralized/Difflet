#!/usr/bin/env python3
"""DiT single-step engine attribution on Trainium (the route A/B direction gate).

Captures a compiled backbone NEFF ``reps`` times with neuron-profile, keeps the
per-run summary metrics, exports the run whose device makespan is closest to
the median as parquet, and computes interval-union attribution in the CARDAN
style: per-class active union, per-class exclusive (unmasked) union, exposed
collective time, and the non-PE non-overlapped share.  The gate metric is

    X = exposed_collective_pct + non_pe_engine_nonoverlapped_pct

as pre-registered in benchmark/flux_cache/dit-step-attribution-protocol-*.json.
This script decides nothing; it reports X and the components.

Usage:
    python scripts/analyze_dit_step_attribution.py \
        --neff ~/.cache/difflet/flux/<hash>/transformer/.../graph.neff \
        --world-size 4 --reps 5 --label flux-1024-tp4 \
        --protocol benchmark/flux_cache/dit-step-attribution-protocol-20260903.json \
        --out /tmp/dit_attr/flux.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

NEURON_PROFILE = "/opt/aws/neuron/bin/neuron-explorer"  # neuron-profile was removed in tools 2.32
NEURON_EXPLORER = "/opt/aws/neuron/bin/neuron-explorer"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], timeout: int) -> str:
    print("  $ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    out = p.stdout.decode("utf-8", "replace")
    if p.returncode != 0 and "numerical error (NaN)" not in out:
        sys.stderr.write(p.stderr.decode("utf-8", "replace")[-3000:])
        p.check_returncode()
    return out


# --- interval algebra (identical to the CARDAN analyzers) -------------------

def merge(intervals):
    out = []
    for start, end in sorted((int(s), int(e)) for s, e in intervals if e > s):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def subtract(left, right):
    left = merge(left)
    right = merge(right)
    out = []
    candidate = 0
    for start, end in left:
        cursor = start
        while candidate < len(right) and right[candidate][1] <= cursor:
            candidate += 1
        probe = candidate
        while probe < len(right) and right[probe][0] < end:
            blocked_start, blocked_end = right[probe]
            if cursor < blocked_start:
                out.append((cursor, min(end, blocked_start)))
            cursor = max(cursor, blocked_end)
            if cursor >= end:
                break
            probe += 1
        if cursor < end:
            out.append((cursor, end))
    return out


def duration_ms(intervals) -> float:
    return sum(e - s for s, e in merge(intervals)) / 1e6


# --- capture ----------------------------------------------------------------

def capture_once(neff: str, workdir: Path, world_size: int, rep: int) -> tuple[dict, str]:
    rep_dir = workdir / f"rep{rep}"
    shutil.rmtree(rep_dir, ignore_errors=True)
    rep_dir.mkdir(parents=True)
    prefix = rep_dir / "profile"
    cmd = [NEURON_PROFILE, "capture", "-n", neff, "-s", str(prefix) + ".ntff",
           "--num-exec", "2", "--profile-nth-exec", "2", "--ignore-exec-errors"]
    if world_size > 1:
        cmd += ["--collectives-workers-per-node", str(world_size),
                "--collectives-profile-id", "0"]
    run(cmd, timeout=1800)
    cands = [f"{prefix}_rank_0_exec_2.ntff", f"{prefix}_exec_2.ntff"]
    ntff = next((c for c in cands if os.path.exists(c)), None)
    if ntff is None:
        raise FileNotFoundError(f"no .ntff produced; looked for {cands}")
    out = run([NEURON_PROFILE, "view", "-n", neff, "-s", ntff,
               "--output-format", "summary-json", "--ignore-nc-buf-usage"], timeout=900)
    metrics = list(json.loads(out).values())[0]
    return metrics, ntff


def export_parquet(neff: str, ntff: str, out_dir: Path) -> Path:
    shutil.rmtree(out_dir, ignore_errors=True)
    run([NEURON_EXPLORER, "view", "-n", neff, "-s", ntff, "--output-format", "parquet",
         "--output-file", str(out_dir), "--ingest-only"], timeout=1800)
    return out_dir


# --- timeline attribution ---------------------------------------------------

def analyze_timeline(profile_dir: Path) -> dict:
    import duckdb  # noqa: WPS433

    def query(table: str, columns: str = "*", where: str = ""):
        parquet = profile_dir / f"{table}.parquet"
        if not parquet.exists():
            raise FileNotFoundError(
                f"{parquet} missing; tables present: {sorted(p.name for p in profile_dir.glob('*.parquet'))}")
        return duckdb.sql(f"SELECT {columns} FROM '{parquet}' {where}").fetchall()

    summary = query("Summary", "total_time, hbm_read_bytes, hbm_write_bytes")[0]
    makespan_ns = float(summary[0]) * 1e9
    makespan_ms = makespan_ns / 1e6

    engines_present = sorted({r[0] for r in query("ActiveTime", "DISTINCT engine")})

    def active(engine: str):
        return query("ActiveTime", "start_ts, end_ts, pcore_idx", f"WHERE engine = '{engine}'")

    gpsimd_dma = query(
        "Instruction", "start_ts, end_ts, pcore_idx",
        "WHERE engine = 'GpSimd' AND (opcode LIKE 'DMA_DIRECT%' OR opcode LIKE 'DMA_INDIRECT%')")
    dma_packets = query("DmaPacket", "start_ts, end_ts")
    dma_service = merge(dma_packets + [(s, e) for s, e, _ in gpsimd_dma])

    classes: dict[str, list] = {}
    for engine in engines_present:
        rows = active(engine)
        if engine == "gpsimd":
            non_dma = []
            for pcore in sorted({r[2] for r in rows}):
                core_active = [(s, e) for s, e, c in rows if c == pcore]
                core_dma = [(s, e) for s, e, c in gpsimd_dma if c == pcore]
                non_dma.extend(subtract(core_active, core_dma))
            classes["gpsimd_non_dma"] = merge(non_dma)
        else:
            classes[engine] = merge((s, e) for s, e, _ in rows)
    classes["dma_service"] = dma_service
    classes["collective"] = merge(
        query("CcOp", "start_ts, end_ts", "WHERE operation NOT IN ('Invalid', 'BARRIER')"))

    active_ms = {k: duration_ms(v) for k, v in classes.items()}
    exclusive_ms = {}
    for name, intervals in classes.items():
        others = merge(iv for other, ivs in classes.items() if other != name for iv in ivs)
        exclusive_ms[name] = duration_ms(subtract(intervals, others))

    tensor = classes.get("tensor", [])
    compute_engine_names = [k for k in classes if k not in ("dma_service", "collective")]
    non_pe_names = [k for k in compute_engine_names if k != "tensor"]
    non_pe_union = merge(iv for k in non_pe_names for iv in classes[k])
    compute_union = merge(iv for k in compute_engine_names for iv in classes[k])

    non_pe_nonoverlapped_ms = duration_ms(subtract(non_pe_union, tensor))
    exposed_cc_ms = duration_ms(subtract(classes["collective"], compute_union))
    exposed_dma_ms = duration_ms(subtract(dma_service, merge(compute_union + classes["collective"])))
    idle_ms = makespan_ms - duration_ms(merge(compute_union + classes["collective"] + dma_service))

    pct = lambda ms: 100.0 * ms / makespan_ms if makespan_ms else float("nan")
    total_leader = max(active_ms, key=active_ms.get)
    exposed_leader = max(exclusive_ms, key=exclusive_ms.get)

    return {
        "engines_present_in_ActiveTime": engines_present,
        "device_makespan_ms": makespan_ms,
        "hbm_read_mb_per_rank": int(summary[1]) / 1e6,
        "hbm_write_mb_per_rank": int(summary[2]) / 1e6,
        "active_union_ms": active_ms,
        "exclusive_union_ms": exclusive_ms,
        "active_union_pct": {k: pct(v) for k, v in active_ms.items()},
        "exclusive_union_pct": {k: pct(v) for k, v in exclusive_ms.items()},
        "tensor_union_pct": pct(active_ms.get("tensor", 0.0)),
        "non_pe_engine_nonoverlapped_pct": pct(non_pe_nonoverlapped_ms),
        "exposed_collective_pct": pct(exposed_cc_ms),
        "exposed_dma_pct": pct(exposed_dma_ms),
        "uncovered_idle_pct": pct(idle_ms),
        "gate_metric_X": pct(non_pe_nonoverlapped_ms) + pct(exposed_cc_ms),
        "critical_path_attribution": (
            total_leader if total_leader == exposed_leader else f"{total_leader}/{exposed_leader}"),
        "critical_path_definition": (
            "leader of both total and exclusive interval-union service; a split label reports "
            "disagreement and is not dependency proof"),
        "definitions": {
            "non_pe_engine_nonoverlapped_pct": "union of all non-tensor compute engines (vector, scalar, gpsimd non-DMA, sync/other) minus tensor-engine active intervals, as % of makespan",
            "exposed_collective_pct": "CcOp intervals (not Invalid/BARRIER) minus the union of all compute engines, as % of makespan",
            "exposed_dma_pct": "DMA service intervals minus compute and collective unions",
            "uncovered_idle_pct": "makespan not covered by any compute, collective, or DMA interval",
        },
    }


def summary_row(m: dict) -> dict:
    g = lambda k: float(m.get(k, 0.0) or 0.0)
    return {
        "total_time_s": g("total_time"),
        "tensor_active_pct": 100 * g("tensor_engine_active_time_percent"),
        "vector_active_pct": 100 * g("vector_engine_active_time_percent"),
        "scalar_active_pct": 100 * g("scalar_engine_active_time_percent"),
        "gpsimd_active_pct": 100 * g("gpsimd_engine_active_time_percent"),
        "sync_active_pct": 100 * g("sync_engine_active_time_percent"),
        "cc_cores_active_pct": 100 * g("cc_cores_instruction_active_time_percent"),
        "cc_op_active_time_s": g("cc_op_active_time"),
        "mfu_pct": 100 * g("mfu_estimated_percent"),
        "mbu_pct": 100 * g("mbu_estimated_percent"),
        "hfu_pct": 100 * g("hfu_estimated_percent"),
        "hbm_read_bytes": g("hbm_read_bytes"),
        "hbm_write_bytes": g("hbm_write_bytes"),
        "dma_active_pct": 100 * (g("hardware_dynamic_dma_active_time_percent")
                                 + g("software_dynamic_dma_active_time_percent")
                                 + g("static_dma_active_time_percent")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neff", required=True)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--label", default="dit")
    ap.add_argument("--protocol", default=None, help="protocol JSON whose sha256 is recorded")
    ap.add_argument("--workdir", default="/tmp/difflet_dit_attr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--analyze-only", default=None,
                    help="skip capture; analyze an existing parquet directory")
    args = ap.parse_args()

    workdir = Path(args.workdir) / args.label
    workdir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "schema": "difflet-dit-step-attribution-result",
        "schema_revision": 1,
        "label": args.label,
        "neff": str(Path(args.neff).resolve()),
        "neff_sha256": sha256_file(Path(args.neff)),
        "world_size": args.world_size,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.protocol:
        proto = json.loads(Path(args.protocol).read_text())
        result["protocol"] = {"path": args.protocol, "sha256": proto.get("sha256"),
                              "study_id": proto.get("study_id")}

    if args.analyze_only:
        result["timeline"] = analyze_timeline(Path(args.analyze_only))
    else:
        reps = []
        for rep in range(args.reps):
            metrics, ntff = capture_once(args.neff, workdir, args.world_size, rep)
            row = {"rep": rep, "ntff": ntff, **summary_row(metrics)}
            reps.append(row)
            print(f"  rep {rep}: total_time={row['total_time_s']*1e3:.2f} ms "
                  f"tensor={row['tensor_active_pct']:.1f}% mfu={row['mfu_pct']:.1f}%", flush=True)
        times = [r["total_time_s"] for r in reps]
        med = statistics.median(times)
        rep_row = min(reps, key=lambda r: abs(r["total_time_s"] - med))
        result["reps"] = reps
        result["medians"] = {k: statistics.median(r[k] for r in reps)
                             for k in reps[0] if isinstance(reps[0][k], (int, float)) and k != "rep"}
        result["representative_rep"] = rep_row["rep"]
        parquet_dir = export_parquet(args.neff, rep_row["ntff"], workdir / "representative_parquet")
        result["representative_parquet"] = str(parquet_dir)
        result["timeline"] = analyze_timeline(parquet_dir)

    t = result["timeline"]
    result["gate"] = {
        "X_exposed_cc_plus_non_pe_pct": t["gate_metric_X"],
        "tensor_union_pct": t["tensor_union_pct"],
        "exposed_collective_pct": t["exposed_collective_pct"],
        "non_pe_engine_nonoverlapped_pct": t["non_pe_engine_nonoverlapped_pct"],
        "rule": "X<20 and tensor_union>=80 -> route A; X>=30 -> route B; else middle (see protocol)",
    }
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["gate"], indent=2))
    print(f"[dit-attr] wrote {args.out}")


if __name__ == "__main__":
    main()
