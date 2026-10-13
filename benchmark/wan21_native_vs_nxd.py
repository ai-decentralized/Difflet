"""Wan2.1 T2V 14B, TorchNeuron (``--backend neuron``) vs NxD AoT (``--backend trainium``).

One protocol for both backends, through the user-facing ``difflet generate``:

* same model + pinned revision, tp=4, 480x832x9, 50 steps, seed, prompt; umT5 and VAE on
  their default (on-device) placement on both backends;
* ``NEURON_RT_VIRTUAL_CORE_SIZE=2`` exported to every launch (compile included: the NxD
  transformer trace reads it to pick ``attention_cte[2]``, and the stage cache key does not
  include it, so each backend gets its own fresh ``--cache-dir``);
* **e2e** = the ``difflet generate`` process wall (spawn -> mp4 written), timed here;
  **cold** = OS page cache dropped right before the launch, compiled artifacts present;
  **warm** = page cache and compile cache warm, new process each time (n=3);
  **first** (neuron only) = empty compile cache: the JIT compile lands in the generate;
* **DiT step** = inter-iteration deltas of the shared ``WanOrchestrator._denoise`` loop,
  stamped after each ``_scheduler_step`` with a device sync, step 0 excluded
  (``benchmark/wan_step_probe/sitecustomize.py``; identical code on both backends).

Usage (one backend per invocation; runs are appended to ``<out>/runs.jsonl``)::

    python benchmark/wan21_native_vs_nxd.py run --backend neuron   --out <dir> --python <venv>
    python benchmark/wan21_native_vs_nxd.py run --backend trainium --out <dir> --python <venv>
    python benchmark/wan21_native_vs_nxd.py report --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "benchmark" / "wan_step_probe"
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
REVISION = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"
PROMPT = "a cinematic shot of a red fox running through a snowy forest"
SHAPE = ["--tp-degree", "4", "--cp-degree", "1", "--height", "480", "--width", "832",
         "--num-frames", "9"]
STEPS = 50
VCS = "2"


def _model_args(backend: str, cache: Path) -> list[str]:
    args = ["--model-id", MODEL_ID, "--revision", REVISION, *SHAPE, "--backend", backend,
            "--cache-dir", str(cache)]
    if backend == "neuron":
        args += ["--exec-mode", "compile"]
    return args


def _env(cache: Path, probe_dir: Path, python: str) -> dict[str, str]:
    env = os.environ.copy()
    # An activated venv: the NxD stack shells out to console scripts (libneuronpjrt-path).
    env["PATH"] = f"{Path(python).parent}{os.pathsep}{env.get('PATH', '')}"
    for name in ("DIFFLET_BACKEND", "DIFFLET_EXEC_MODE", "TORCH_NEURONX_NEFF_CACHE_DIR",
                 "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "TORCH_NEURONX_HLO_CACHE_DIR",
                 "NEURON_RT_NUM_CORES", "NEURON_LOGICAL_NC_CONFIG"):
        env.pop(name, None)
    env.update({
        "NEURON_RT_VIRTUAL_CORE_SIZE": VCS,
        "HF_HUB_OFFLINE": "1",
        "DIFFLET_COMPILE_CACHE": str(cache),
        "DIFFLET_NEURON_PG_TIMEOUT_S": "3600",
        "DIFFLET_WAN_PROBE_DIR": str(probe_dir),
        "PYTHONPATH": f"{PROBE}{os.pathsep}{REPO}",
    })
    return env


def _count_files(root: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(root)) if root.exists() else 0


def _cached_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("Cached:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def _drop_page_cache() -> None:
    subprocess.run(["sync"], check=True)
    subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)


def _launch(out: Path, python: str, backend: str, name: str, command: str, extra: list[str],
            *, guidance: float | None, drop: bool) -> dict:
    run_dir = out / backend / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = out / backend / "cache"
    probe_dir = run_dir / "probe"
    cmd = [python, "-m", "difflet.cli.main", command, *_model_args(backend, cache), *extra]
    files_before = _count_files(cache)
    if drop:
        _drop_page_cache()
    cached_before = _cached_gib()
    start_wall = time.time()
    start = time.perf_counter()
    with open(run_dir / "log.txt", "w") as log:
        rc = subprocess.run(cmd, cwd=run_dir, env=_env(cache, probe_dir, python), stdout=log,
                            stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - start
    probes = [json.loads(p.read_text()) for p in sorted(probe_dir.glob("probe-*.json"))]
    record = {
        "backend": backend, "name": name, "command": command, "guidance": guidance,
        "drop_page_cache": drop, "page_cache_gib_at_start": round(cached_before, 1),
        "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_wall)),
        "wall_s": round(wall, 2), "returncode": rc,
        "new_cache_files": _count_files(cache) - files_before,
        "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
        "argv": cmd, "probes": probes,
    }
    with open(out / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"[bench] {backend}/{name}: rc={rc} wall={wall:.1f}s new_cache_files="
          f"{record['new_cache_files']} {_step_line(record)}", flush=True)
    if rc != 0:
        raise SystemExit(f"{backend}/{name} failed (rc={rc}); see {run_dir / 'log.txt'}")
    return record


def _generate_args(guidance: float) -> list[str]:
    return ["--steps", str(STEPS), "--guidance-scale", str(guidance), "--seed", "42",
            "--prompt", PROMPT, "--output", "out.mp4"]


def _done(out: Path, backend: str) -> set[str]:
    path = out / "runs.jsonl"
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["name"] for r in rows if r["backend"] == backend and r["returncode"] == 0}


def cmd_run(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    backend = args.backend
    plan: list[tuple[str, str, list[str], float | None, bool]] = []
    if backend == "trainium":
        plan.append(("compile", "compile", [], None, False))
    else:
        # The JIT compile lands in the first generate on an empty cache.
        plan.append(("first-g5", "generate", _generate_args(5.0), 5.0, False))
    plan.append(("cold-g5", "generate", _generate_args(5.0), 5.0, True))
    plan += [(f"warm{i}-g5", "generate", _generate_args(5.0), 5.0, False) for i in (1, 2, 3)]
    plan += [(f"warm{i}-g1", "generate", _generate_args(1.0), 1.0, False) for i in (1, 2)]
    done = _done(out, backend)
    for name, command, extra, guidance, drop in plan:
        if name in done:
            print(f"[bench] {backend}/{name}: already recorded, skipping", flush=True)
            continue
        _launch(out, args.python, backend, name, command, extra, guidance=guidance, drop=drop)
    print(f"[bench] {backend}: ALL DONE", flush=True)


# ------------------------------------------------------------------ report

def _steps_ms(record: dict) -> tuple[list[float], list[float]]:
    """(rank-0 / single-process step series, per-step max over ranks), step 0 excluded, ms."""
    series = [p["step_s"][1:] for p in record["probes"] if p.get("step_s")]
    if not series:
        return [], []
    by_rank = sorted(record["probes"], key=lambda p: int(p.get("rank") or 0))
    first = [s * 1e3 for s in by_rank[0]["step_s"][1:]]
    rank_max = [max(col) * 1e3 for col in zip(*series)]
    return first, rank_max


def _step_line(record: dict) -> str:
    first, rank_max = _steps_ms(record)
    if not first:
        return ""
    return (f"step median {statistics.median(first):.1f} ms (rank-max "
            f"{statistics.median(rank_max):.1f}), n={len(first)}")


def _stage(record: dict, key: str) -> float | None:
    values = [p[key] for p in record["probes"] if key in p]
    if not values:
        return None
    return max(values) if not isinstance(values[0], list) else max(sum(v) for v in values)


def _fmt(values: list[float], unit: str = "s") -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return (f"{statistics.mean(values):.1f} (median {statistics.median(values):.1f}; "
            f"{min(values):.1f}-{max(values):.1f}, n={len(values)})")


def cmd_report(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve()
    rows = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()
            if line.strip()]
    by = {(r["backend"], r["name"]): r for r in rows if r["returncode"] == 0}
    lines = ["| metric | TorchNeuron (`--backend neuron`) | NxD AoT (`--backend trainium`) |",
             "|---|---:|---:|"]

    def cell(backend: str, fn) -> str:
        try:
            return fn(backend)
        except (KeyError, ValueError, statistics.StatisticsError, TypeError):
            return "—"

    def wall(name):
        return lambda b: f"{by[(b, name)]['wall_s']:.1f}"

    def warm(g):
        return lambda b: _fmt([by[(b, f'warm{i}-g{g}')]['wall_s'] for i in (1, 2, 3)
                               if (b, f'warm{i}-g{g}') in by])

    def step(g, rank_max=False):
        def f(b):
            pooled = []
            for i in (1, 2, 3):
                r = by.get((b, f"warm{i}-g{g}"))
                if r:
                    pooled += _steps_ms(r)[1 if rank_max else 0]
            return (f"{statistics.median(pooled):.1f} (mean {statistics.mean(pooled):.1f}, "
                    f"n={len(pooled)})")
        return f

    def stage(name, key):
        return lambda b: f"{_stage(by[(b, name)], key):.2f}"

    lines += [
        "| one-time compile (s) | " + cell("neuron", lambda b: "in first generate: "
            f"{by[(b, 'first-g5')]['wall_s'] - by[(b, 'warm1-g5')]['wall_s']:.1f} "
            "(first − warm1)") + " | " + cell("trainium", wall("compile")) + " |",
        "| e2e first-ever generate, empty compile cache (s) | " + cell("neuron", wall("first-g5"))
            + " | n/a (AoT: compile is a separate step) |",
        "| **e2e cold** (page cache dropped, compiled) g=5 (s) | " + cell("neuron", wall("cold-g5"))
            + " | " + cell("trainium", wall("cold-g5")) + " |",
        "| **e2e warm** g=5 (s) | " + cell("neuron", warm(5)) + " | " + cell("trainium", warm(5))
            + " |",
        "| e2e warm g=1 (s) | " + cell("neuron", warm(1)) + " | " + cell("trainium", warm(1)) + " |",
        "| **DiT step median** g=5 (ms) | " + cell("neuron", step(5)) + " | "
            + cell("trainium", step(5)) + " |",
        "| DiT step rank-max median g=5 (ms) | " + cell("neuron", step(5, True)) + " | "
            + cell("trainium", step(5, True)) + " |",
        "| **DiT step median** g=1 (ms) | " + cell("neuron", step(1)) + " | "
            + cell("trainium", step(1)) + " |",
        "| denoise loop, warm1 g=5 (s) | " + cell("neuron", stage("warm1-g5", "denoise_s"))
            + " | " + cell("trainium", stage("warm1-g5", "denoise_s")) + " |",
        "| prompt encode, warm1 g=5 (s) | " + cell("neuron", stage("warm1-g5", "encode_s"))
            + " | " + cell("trainium", stage("warm1-g5", "encode_s")) + " |",
    ]
    detail = ["", "| backend | run | start (UTC) | wall (s) | new cache files | page cache at "
              "start (GiB) | step median (ms) | VCS seen by denoise process(es) |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        first, _ = _steps_ms(r)
        vcs = sorted({str(p.get("env", {}).get("NEURON_RT_VIRTUAL_CORE_SIZE"))
                      for p in r["probes"]}) or ["—"]
        detail.append(
            f"| {r['backend']} | {r['name']} | {r['start_utc']} | {r['wall_s']:.1f} | "
            f"{r['new_cache_files']} | {r['page_cache_gib_at_start']} | "
            f"{statistics.median(first) if first else float('nan'):.1f} | "
            f"{','.join(vcs)} ({len(r['probes'])} proc) |")
    text = "\n".join(lines + detail) + "\n"
    (out / "summary.md").write_text(text)
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--backend", choices=("neuron", "trainium"), required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--python", required=True, help="the backend's venv interpreter")
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report")
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
