#!/usr/bin/env python3
"""Per-step latency for every feasible 4-core Flux parallel config.

Methodology is ``benchmark/step_realloop.py``'s, verbatim: build the pipeline
in-process exactly as the CLI orchestrator does (compiled NEFF,
``skip_compile=True``), wrap the per-step DiT call so a synchronized
``perf_counter`` lands after each step, run a real generate, and report the
inter-step deltas with step 0 excluded -- the same quantity the H100 reference
measures. One measurement process per config: each build loads a ~34 GB
pipeline onto the 4 cores, so builds cannot share a process.

Each config runs ``difflet compile`` (subprocess; cache-hit fast when the
artifact exists) -> one discarded warm-up generate -> one timed generate.

DP rows are derived, not faked: a dp2tp2 replica loads and executes the same
tp2 artifact as a tp2 pipeline (verify_cli's "compile resolves to the same
dp=1 tp2 artifact both replicas load"), so its per-STEP latency is the tp2
measurement; what dp adds is process routing, which lives in e2e, not in the
DiT step. The benchmark JSON for dp rows records that derivation explicitly.

Writes seeder-compatible ``benchmark/trn2/flux_<label>.json`` (tp4 already has
the committed flux_1_dev.json anchor; it is re-measured as a cross-check but
not re-seeded). Then:

    python scripts/seed_planner_measurements.py --device trn2
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL_ID = "black-forest-labs/FLUX.1-dev"
REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
PROMPT = "a cat sitting on a bench"
STEPS = 28
HEIGHT, WIDTH = 1024, 1024
DEVICE = "trn2.3xlarge / 4 NeuronCores / 96 GB/device"

# label -> DiffletParallelConfig kwargs. Exactly the configs that survive
# feasibility on a 4-core host after the D35 memory rule removes the 4-copy
# (~135 GB) candidates. tp2/tp2sp exist to measure dp2tp2/dp2tp2sp's step.
CONFIGS: dict[str, dict] = {
    "tp4": {"tp_degree": 4},
    "tp4sp": {"tp_degree": 4, "sp_enabled": True},
    "tp2cp2": {"tp_degree": 2, "cp_degree": 2},
    "tp2cp2ring": {"tp_degree": 2, "cp_degree": 2, "cp_mode": "ring"},
    "tp2cp2ulysses": {"tp_degree": 2, "cp_degree": 2, "cp_mode": "ulysses"},
    "tp2": {"tp_degree": 2},
    "tp2sp": {"tp_degree": 2, "sp_enabled": True},
}

# dp rows seeded from the identical artifact/step graph of their tp base.
DERIVED_DP = {"dp2tp2": "tp2", "dp2tp2sp": "tp2sp"}

ALREADY_ANCHORED = {"tp4"}  # committed flux_1_dev.json owns this key

LOG_ROOT = REPO / "artifacts" / "flux_parallel_sweep"
BENCH_DIR = REPO / "benchmark" / "trn2"


# ------------------------------------------------------------------ worker


def measure_one(label: str) -> int:
    """In-process realloop measurement of one config; prints JSON and exits."""

    import torch
    from difflet.pipeline.difflet_pipeline import DiffletPipeline
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    parallel = DiffletParallelConfig(**CONFIGS[label])
    cache = Path("~/.cache/difflet").expanduser()
    print(f"[sweep:{label}] building pipeline {parallel}...", flush=True)
    t0 = time.perf_counter()
    pipe = DiffletPipeline.from_pretrained(
        MODEL_ID, model_type="flux", parallel=parallel, dtype=torch.bfloat16,
        height=HEIGHT, width=WIDTH, compile_cache_dir=str(cache),
        revision=REVISION, skip_compile=True)
    dit_cls = type(pipe.app.pipe.transformer)  # NeuronFluxBackboneApplication
    print(f"[sweep:{label}] pipeline ready in {time.perf_counter() - t0:.1f}s", flush=True)

    stamps: list[float] = []
    orig = dit_cls.__call__

    def timed(self, *a, **k):
        r = orig(self, *a, **k)
        stamps.append(time.perf_counter())  # host tensors returned -> synced
        return r

    gen_kwargs = dict(
        prompt=PROMPT, num_inference_steps=STEPS, height=HEIGHT, width=WIDTH,
        guidance_scale=3.5, generator=torch.Generator().manual_seed(42),
        output_type="pt")

    dit_cls.__call__ = timed
    try:
        pipe(**gen_kwargs)          # warm-up: page cache + NEFF dispatch
        stamps.clear()
        t1 = time.perf_counter()
        result = pipe(**gen_kwargs)  # timed
        gen_s = time.perf_counter() - t1
    finally:
        dit_cls.__call__ = orig

    deltas = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    finite = None
    try:
        img = getattr(result, "images", None)
        if img is None:
            img = result[0]
        if isinstance(img, torch.Tensor):
            finite = bool(torch.isfinite(img).all())
    except Exception as exc:  # output inspection must never discard the timing
        print(f"[sweep:{label}] finite-check skipped: {exc}", flush=True)
    print("SWEEP_RESULT " + json.dumps({
        "label": label,
        "mean": statistics.fmean(deltas),
        "median": statistics.median(deltas),
        "min": min(deltas),
        "max": max(deltas),
        "p90": sorted(deltas)[int(0.9 * (len(deltas) - 1))],
        "n": len(deltas),
        "generate_wall_s": gen_s,
        "finite": finite,
    }))
    return 0


# -------------------------------------------------------------- orchestrator


def _run(cmd: list[str], log: Path, timeout: float) -> float:
    t0 = time.perf_counter()
    with log.open("w") as fh:
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        raise RuntimeError(f"failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(tail))
    return time.perf_counter() - t0


def _stats(d: dict) -> dict:
    return {k: d[k] for k in ("mean", "median", "min", "max", "p90", "n")} | {
        "generate_wall_s": d["generate_wall_s"]}


def _world_size(kw: dict) -> int:
    return kw.get("tp_degree", 1) * kw.get("cp_degree", 1)


def _write_benchmark(label: str, row: dict, *, derived: bool = False) -> None:
    """Emit (or re-emit) the seeder-compatible benchmark JSON for one label."""

    if label in ALREADY_ANCHORED:
        return
    base = DERIVED_DP[label] if derived else label
    parallel = dict(CONFIGS[base]) | ({"dp_degree": 2} if derived else {})
    payload = {
        "model_id": MODEL_ID,
        "model_type": "flux",
        "backend": "trainium",
        "device": DEVICE,
        "dtype": "bf16",
        "parallel": {
            "tp_degree": parallel.get("tp_degree", 1),
            "cp_degree": parallel.get("cp_degree", 1),
            "cp_mode": parallel.get("cp_mode", "gather_kv"),
            "cfg_parallel_enabled": False,
            "sp_enabled": parallel.get("sp_enabled", False),
            "dp_degree": parallel.get("dp_degree", 1),
        },
        "shape": {"height": HEIGHT, "width": WIDTH, "num_frames": None},
        "steps": STEPS,
        "compile_seconds": row.get("compile_seconds"),
        "step_latency": _stats(row),
        "e2e_warm": {"median": row["generate_wall_s"], "n": 1},
        "notes": [
            f"per-step measured by scripts/flux_parallel_sweep.py on trn2.3xlarge "
            f"(2026-08-20) using the step_realloop "
            f"method: inter-step deltas of a real 28-step generate wrapping "
            f"NeuronFluxBackboneApplication.__call__, synced, step 0 excluded, "
            f"one discarded warm-up generate. output finite={row['finite']}."
        ],
    }
    if derived:
        payload["notes"].append(
            f"DERIVED per-step: a {label} replica loads and executes the same "
            f"{base} artifact (compile-once-load-k), so its DiT-step latency is "
            f"the {base} measurement; dp adds only process routing, which is e2e "
            f"overhead, not step overhead.")
    out = BENCH_DIR / f"flux_{label}.json"
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"    wrote {out.relative_to(REPO)}", flush=True)


def _recover_from_logs() -> dict[str, dict]:
    """Re-populate results from a previous (interrupted) sweep's measure logs.

    A SWEEP_RESULT line is printed only after a measurement completed, so its
    presence in <label>_measure.log means that config's number is good.
    """

    results: dict[str, dict] = {}
    for label in CONFIGS:
        log = LOG_ROOT / f"{label}_measure.log"
        if not log.exists():
            continue
        lines = [
            l for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.startswith("SWEEP_RESULT ")
        ]
        if not lines:
            continue
        payload = json.loads(lines[-1][len("SWEEP_RESULT "):])
        payload.setdefault("compile_seconds", None)
        results[label] = payload
        print(f"    [recovered] {label}: median {payload['median'] * 1000:.1f} ms", flush=True)
        _write_benchmark(label, payload)
    return results


def sweep(only: list[str] | None = None) -> int:
    import os

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    results = _recover_from_logs()
    labels = [l for l in CONFIGS if not only or l in only]

    for label in labels:
        if label in results:
            print(f">>> [{label}] already measured, skipping", flush=True)
            continue
        flags = []
        kw = CONFIGS[label]
        flags += ["--tp-degree", str(kw["tp_degree"])]
        if kw.get("cp_degree", 1) > 1:
            flags += ["--cp-degree", str(kw["cp_degree"])]
            if kw.get("cp_mode", "gather_kv") != "gather_kv":
                flags += ["--cp-mode", kw["cp_mode"]]
        if kw.get("sp_enabled"):
            flags.append("--sp")

        print(f">>> [{label}] compile ...", flush=True)
        compile_wall = _run(
            ["difflet", "compile", "--model-id", MODEL_ID, "--revision", REVISION]
            + flags + ["--height", str(HEIGHT), "--width", str(WIDTH)],
            LOG_ROOT / f"{label}_compile.log", timeout=3 * 3600)
        print(f"    compile wall {compile_wall:.0f}s "
              f"({'cache hit' if compile_wall < 180 else 'cold'})", flush=True)

        print(f">>> [{label}] measure (in-process realloop) ...", flush=True)
        log = LOG_ROOT / f"{label}_measure.log"
        t0 = time.perf_counter()
        # A world smaller than the device must pin its cores explicitly: the
        # in-process load otherwise bootstraps a 4-rank group and the ranks
        # beyond world_size never find a root (the DP router does the same
        # pinning for its tp2 workers).
        env = None
        world = _world_size(kw)
        if world < 4:
            env = dict(
                os.environ,
                NEURON_RT_VISIBLE_CORES=",".join(str(i) for i in range(world)),
            )
        with log.open("w") as fh:
            proc = subprocess.run(
                [sys.executable, __file__, "--measure", label],
                stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=2 * 3600,
                env=env)
        if proc.returncode != 0:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            raise RuntimeError(f"[{label}] measure failed\n" + "\n".join(tail))
        text = log.read_text(encoding="utf-8", errors="replace")
        line = next(l for l in text.splitlines() if l.startswith("SWEEP_RESULT "))
        payload = json.loads(line[len("SWEEP_RESULT "):])
        payload["compile_seconds"] = compile_wall
        payload["measure_wall_s"] = time.perf_counter() - t0
        results[label] = payload
        print(f"    [{label}] median step {payload['median'] * 1000:.1f} ms "
              f"(n={payload['n']}, finite={payload['finite']})", flush=True)
        _write_benchmark(label, payload)  # persist immediately: crashes lose nothing

    # DP rows: same artifact and per-step graph as their tp base.
    for dp_label, base in DERIVED_DP.items():
        if only and dp_label not in only:
            continue
        if base not in results:
            continue
        row = dict(results[base])
        row["label"] = dp_label
        results[dp_label] = row
        _write_benchmark(dp_label, row, derived=True)

    summary = LOG_ROOT / "sweep_summary.json"
    summary.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nsummary -> {summary}")
    for label in sorted(results, key=lambda k: results[k]["median"]):
        print(f"  {label:<16} {results[label]['median'] * 1000:8.1f} ms/step")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measure", metavar="LABEL", help="worker: measure this config in-process")
    p.add_argument("--only", nargs="*", help="restrict the sweep to these labels")
    args = p.parse_args()
    if args.measure:
        return measure_one(args.measure)
    return sweep(args.only or None)


if __name__ == "__main__":
    raise SystemExit(main())
