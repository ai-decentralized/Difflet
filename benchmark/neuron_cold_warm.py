"""Wan2.1 T2V 14B on the neuron backend, TP4, 480x832x9, 50 steps: compile time, e2e cold,
e2e warm and DiT step time, per guidance value. The driver; it never touches the device.

Per guidance value (5.0 first, the headline; then 1.0, the trn2-comparable row) it runs, one
device job at a time, each launch after the cores-free check:

1. ``script-cold``: a fresh ``DIFFLET_COMPILE_CACHE`` (empty ``neuron/{neff,neff_local,hlo}``)
   and one torchrun launch of ``benchmark/neuron_wan_bench.py`` with three generates (the
   first is the cold number, the others the resident rows);
2. ``script-warm1..3``: the same launch three more times on that cache (warm: per-rank
   ``TotalCompilations == 0 and PersistentHits >= 1``, no new ``.neff``);
3. at guidance 5.0 only, ``cli-w4check``: one warm ``difflet generate`` with W4's exact
   parameters (the W5 section-4 command), which must reproduce W4's latents and mp4 (D80); a
   mismatch stops the job;
4. ``cli-warm1..3``: the user-facing ``difflet generate --backend neuron`` on the same cache;
5. ``cli-truecold`` (when ``sudo -n`` works): the page cache dropped first, NEFF cache warm,
   the trn2 definition of a cold start;
6. ``cli-cold``: ``difflet generate`` on its own fresh cache (the user-facing e2e cold; its
   block compile is a second, intentional compile sample, not the compile-time number).

Every launch records the host before and after (``free -g``, ``/proc/meminfo``, NEFF / HLO /
``/tmp`` cache file lists, the sysfs throttle strings, neuronx-cc version, thread counts, git
HEAD and the benchmark files' sha256) under ``<record-dir>/runs/g<guidance>/<launch>/``.
Hard failures (a non-zero exit, a fallback, an unwarmed shape, ranks or iterations with
different latents, a broken counter rule on a warm launch, the W4 hashes) stop the job;
statistical checks (step tail and drift, the 2 % resident rule) are flagged, not enforced.
At the end (or at a stop) it assembles one ``BenchResult`` per guidance value and writes
``benchmark/trn2/wan_2_1_neuron.json`` and ``.md`` (guidance 5.0 first).

    PYTHONSAFEPATH=1 PYTHONPATH=<worktree> <venv>/bin/python -m benchmark.neuron_cold_warm \\
        --record-dir /home/ubuntu/plans/wan21/records/W6

``--dry-run`` prints the plan and the commands; ``--assemble-only`` re-renders the results
from the recorded launches; ``--skip-existing`` resumes after a stop.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark import report
from benchmark.adapters import neuron as na
from benchmark.harness import BenchResult, OutputInfo, Stats
from benchmark.models import MATRIX

SLUG = "wan_2_1_neuron"
GUIDANCES = (5.0, 1.0)
HEADLINE_GUIDANCE = 5.0
BENCH_FILES = ("benchmark/neuron_cold_warm.py", "benchmark/neuron_wan_bench.py",
               "benchmark/adapters/neuron.py", "benchmark/harness.py", "benchmark/report.py",
               "benchmark/models.py", "tests/numerical/neuron_parity.py")
#: W4's 50-step final run, reproduced by W5's CLI cold/warm runs ($REC/W5/SUMMARY.txt).
W4_EXPECTED = {
    "prompt": "a cinematic shot of a red fox running through a snowy forest",
    "seed": 42, "steps": 50, "guidance_scale": 5.0,
    "latents_file_sha256": "e639f3146c3619406267c317c9798b8ead7300535c7835a7d7c7c0de54769028",
    "latents_tensor_sha256":
        "8e5eaffed4cf6f3a79a94dc11664c4f6b49767375fc6f5c451a39330bed990f6",
    "mp4_sha256": "a9f9004fc1343eab167e70f4522f382243f40b7b83765b726e01d02cb01b7f34",
}
#: benchmark/trn2/wan_2_1.json (NxD AoT, 20 steps, guidance 1.0).
TRN2 = {"compile": "7879 (incl. VAE 6098; transformer 255)", "load": "343 cold / 28.4 warm",
        "text_encoder_load": "94.4 cold / 8.3 warm (umT5 NEFF load)",
        "e2e": "394 (page cache dropped) / 56.4 (n=3)", "step": "554.8 (isolated timer)",
        "throughput": "1.80", "peak": "null", "output": "(1,3,9,480,832)"}
BLOCK_NEFF_MIN_BYTES = 1 << 20
BENCH_TIMEOUT_S = {"cold": 3600.0, "warm": 2400.0}
CLI_TIMEOUT_S = 2400.0


# ------------------------------------------------------------------------------- the plan


@dataclasses.dataclass(frozen=True)
class Launch:
    name: str
    kind: str              # "bench" | "cli" | "cli-w4"
    guidance: float
    cache: str             # which cache root of this guidance: "main" | "cli-cold"
    expect: str            # "cold" | "warm"
    fresh_cache: bool = False
    drop_page_cache: bool = False


def build_plan(guidances: Sequence[float] = GUIDANCES, *, warm_launches: int = 3,
               cli_warm: int = 3, true_cold: bool = True,
               w4_guidance: float | None = HEADLINE_GUIDANCE) -> list[Launch]:
    plan: list[Launch] = []
    for g in guidances:
        plan.append(Launch("script-cold", "bench", g, "main", "cold", fresh_cache=True))
        plan += [Launch(f"script-warm{i}", "bench", g, "main", "warm")
                 for i in range(1, warm_launches + 1)]
        if w4_guidance is not None and g == w4_guidance:
            plan.append(Launch("cli-w4check", "cli-w4", g, "main", "warm"))
        plan += [Launch(f"cli-warm{i}", "cli", g, "main", "warm") for i in range(1, cli_warm + 1)]
        if true_cold:
            plan.append(Launch("cli-truecold", "cli", g, "main", "warm", drop_page_cache=True))
        plan.append(Launch("cli-cold", "cli", g, "cli-cold", "cold", fresh_cache=True))
    return plan


def gkey(g: float) -> str:
    return f"g{g:.1f}"


def run_dir(record_dir: Path, launch: Launch) -> Path:
    return record_dir / "runs" / gkey(launch.guidance) / launch.name


def cache_root(record_dir: Path, launch: Launch) -> Path:
    return record_dir / "caches" / f"{gkey(launch.guidance)}-{launch.cache}"


#: ``--steps`` (smoke runs only; the benchmark is the MATRIX entry's 50).
STEPS_OVERRIDE: int | None = None


def config_for(g: float):
    cfg = dataclasses.replace(MATRIX[SLUG], guidance_scale=g)
    return cfg if STEPS_OVERRIDE is None else dataclasses.replace(cfg, steps=STEPS_OVERRIDE)


def w5_command(python: str, *, output, cache_dir, work_dir) -> list[str]:
    """W5 HOW_TO_VERIFY section 4's command line (W4's parameters) with our paths."""
    e = W4_EXPECTED
    return [python, "-m", "difflet.cli.main", "generate", "--backend", "neuron",
            "--model-id", MATRIX[SLUG].model_id, "--revision", MATRIX[SLUG].revision,
            "--tp-degree", "4", "--height", "480", "--width", "832", "--num-frames", "9",
            "--steps", str(e["steps"]), "--guidance-scale", f"{e['guidance_scale']:g}",
            "--seed", str(e["seed"]), "--host-vae", "--prompt", e["prompt"],
            "--output", os.fspath(output), "--cache-dir", os.fspath(cache_dir),
            "--work-dir", os.fspath(work_dir), "--keep-work-dir"]


# ------------------------------------------------------------------------- host evidence


def _run_text(cmd: Sequence[str], timeout: float = 60.0) -> str:
    try:
        out = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
        return (out.stdout + out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def meminfo() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "MemFree", "Cached", "Buffers"):
            values[f"{key}_gib"] = int(value.split()[0]) / 2**20
    return values


def file_list(directory: Path, pattern: str = "*") -> dict[str, int]:
    if not directory.is_dir():
        return {}
    out = {}
    for path in directory.rglob(pattern):
        if path.is_file():
            try:
                out[str(path.relative_to(directory))] = path.stat().st_size
            except OSError:
                continue
    return out


def throttle_strings() -> dict[str, str]:
    root = Path("/sys/class/neuron_device/neuron0")
    values = {}
    for core in sorted(root.glob("neuron_core*")):
        try:
            values[core.name] = (core / "stats/hardware/ham_throttle/throttle").read_text().strip()
        except OSError as exc:
            values[core.name] = f"unreadable: {exc}"
    return values


def host_state(root: Path) -> dict[str, Any]:
    dirs = na.cache_dirs(root)
    return {
        "epoch": time.time(),
        "free_g": _run_text(["free", "-g"]),
        "meminfo": meminfo(),
        "neff": file_list(dirs["neff"], "*.neff"),
        "cache_files": len(file_list(root)),
        "neff_local_files": len(file_list(dirs["neff_local"])),
        "hlo_files": len(file_list(dirs["hlo"])),
        "tmp_neff_cache_files": len(file_list(Path("/tmp/neff_cache"))),
        "tmp_hlo_cache_files": len(file_list(Path("/tmp/hlo_cache"))),
        "throttle": throttle_strings(),
    }


def provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return _run_text(["git", "-C", str(na.REPO_ROOT), *args])

    hashes = {}
    for rel in BENCH_FILES:
        path = na.REPO_ROOT / rel
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {
        "head": git("rev-parse", "HEAD"),
        "worktree": str(na.REPO_ROOT),
        "status": git("status", "--porcelain"),
        "difflet_diff_vs_head": git("diff", "--stat", "HEAD", "--", "difflet"),
        "files_sha256": hashes,
        "neuronx_cc": _run_text([str(Path(sys.executable).parent / "neuronx-cc"), "--version"]
                                ).splitlines()[0:1],
        "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS"),
        "python": sys.executable,
    }


def wait_cores_free(timeout: float = 120.0) -> dict[str, Any]:
    """neuron-ls lists no process and the D9 ``ps`` pattern prints nothing; poll briefly
    (a launch's ranks can take a moment to leave the device after torchrun returns)."""
    deadline = time.monotonic() + timeout
    own = {os.getpid(), os.getppid()}
    while True:
        ls = _run_text(["/opt/aws/neuron/bin/neuron-ls", "-j"])
        try:
            procs = na.device_processes(ls)
        except ValueError:
            procs = [f"neuron-ls output not JSON: {ls[:200]}"]
        busy = na.busy_processes(_run_text(["ps", "-eo", "pid,args"]), sorted(own))
        if not procs and not busy:
            return {"neuron_ls_processes": [], "ps_busy": []}
        if time.monotonic() > deadline:
            raise RuntimeError(f"cores not free: neuron-ls {procs}, ps {busy}")
        time.sleep(5)


def drop_page_cache() -> dict[str, Any]:
    before = meminfo()
    out = subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                         capture_output=True, text=True)
    return {"ok": out.returncode == 0, "stderr": out.stderr.strip(), "before": before,
            "after": meminfo()}


def sudo_available() -> bool:
    return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


# ---------------------------------------------------------------------------- one launch


def new_neffs(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {k: v for k, v in after.items() if k not in before}


def judge_bench(launch: Launch, summary: Mapping[str, Any] | None, run: Mapping[str, Any],
                added: Mapping[str, int], reference_hash: str | None) -> list[dict[str, Any]]:
    """Checks of one bench launch: ``hard`` ones stop the job when they fail."""
    checks: list[dict[str, Any]] = []

    def add(name, ok, detail, hard=True):
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "hard": hard})

    add("exit", run["exit_code"] == 0 and not run["timed_out"],
        f"exit {run['exit_code']} timed_out={run['timed_out']}")
    if summary is None:
        return checks
    its = summary["iterations"]
    add("fallbacks", summary["checks"]["load_fallbacks"]
        and all(it["checks"]["fallbacks"] for it in its),
        f"load {[r['load_fallbacks'] for r in summary['ranks']]}, generates "
        f"{[it['fallbacks_by_rank'] for it in its]}")
    add("unwarmed_shapes", all(it["checks"]["unwarmed_shapes"] for it in its),
        f"{[it['unwarmed_by_rank'] for it in its]}")
    add("compiled_blocks", summary["checks"]["compiled_blocks"],
        f"{[r['compiled_blocks'] for r in summary['ranks']]}")
    add("loop_shape", all(it["checks"]["stamps"] and it["checks"]["deltas"]
                          and it["checks"]["forwards"] for it in its),
        "one stamp per iteration, steps-1 deltas, forwards = steps x (2 if g > 1)")
    add("latents_equal_on_ranks", all(it["checks"]["latents_equal_on_ranks"] for it in its),
        f"{[it['latents_sha256'] for it in its]}")
    add("latents_equal_across_iterations", summary["checks"]["latents_equal_across_iterations"],
        f"{[it['latents_sha256'] for it in its]}")
    if reference_hash is not None:
        add("latents_equal_to_cold_launch", its[0]["latents_sha256"] == reference_hash,
            f"{its[0]['latents_sha256']} vs {reference_hash}")
    add("mp4_written", all(it["checks"]["mp4_written"] for it in its),
        f"{[it['mp4_bytes'] for it in its]} bytes")
    add("hlo_locks_in_cache", summary["checks"]["hlo_locks_in_cache"],
        f"{[r['hlo_lock_dir_at_import'] for r in summary['ranks']]}")
    ok, text = na.counter_rule(summary, launch.expect, len(added))
    add(f"counter_rule_{launch.expect}", ok, text)
    no_compile = all(it["no_compile"] for it in its)
    add("no_compile_in_generates", no_compile,
        f"generate TotalCompilations by iteration and rank "
        f"{[[c['TotalCompilations'] for c in it['counters_by_rank']] for it in its]}",
        hard=launch.expect == "warm")
    for it in its:
        sc = it["step_checks"]
        add(f"step_series_it{it['index']}", sc["ok"], sc["detail"], hard=False)
    if summary["resident_within_2pct"] is not None:
        add("resident_medians_within_2pct", summary["resident_within_2pct"],
            f"step medians {[round(it['step']['median'] * 1000, 2) for it in its]} ms, spread "
            f"{summary['resident_step_median_spread'] * 100:.2f}%", hard=False)
    if launch.guidance == W4_EXPECTED["guidance_scale"]:
        add("latents_equal_w4", its[0]["latents_sha256"] == W4_EXPECTED["latents_tensor_sha256"],
            f"{its[0]['latents_sha256']} vs W4 {W4_EXPECTED['latents_tensor_sha256']}",
            hard=False)
        add("mp4_equal_w4", its[0]["mp4_sha256"] == W4_EXPECTED["mp4_sha256"],
            f"{its[0]['mp4_sha256']} vs W4 {W4_EXPECTED['mp4_sha256']}", hard=False)
    return checks


def judge_cli(launch: Launch, cli: Mapping[str, Any], added_neff: Mapping[str, int],
              hlo_delta: int, reference_hash: str | None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name, ok, detail, hard=True):
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "hard": hard})

    run = cli["launch"]
    add("exit", run["exit_code"] == 0 and not run["timed_out"],
        f"exit {run['exit_code']} timed_out={run['timed_out']}")
    for name, ok, detail in cli["checks"]:
        add(name, ok, detail)
    if launch.expect == "warm":
        add("file_rule_warm", not added_neff and hlo_delta == 0,
            f"new .neff {len(added_neff)}, new HLO-cache files {hlo_delta} (the CLI prints no "
            "counters; a warm launch adds nothing to the row's cache)")
    else:
        add("file_rule_cold", len(added_neff) >= 1, f"new .neff {len(added_neff)}")
    inspected = cli["inspect"]
    tensor = inspected.get("tensor_sha256")
    if reference_hash is not None:
        add("latents_equal_to_script", tensor == reference_hash,
            f"{tensor} vs script {reference_hash}", hard=launch.kind == "cli-w4")
    if launch.kind == "cli-w4":
        mp4 = (inspected.get("mp4") or {}).get("sha256")
        add("w4_latents_file", inspected.get("file_sha256") == W4_EXPECTED["latents_file_sha256"],
            f"{inspected.get('file_sha256')} vs {W4_EXPECTED['latents_file_sha256']}")
        add("w4_latents_tensor", tensor == W4_EXPECTED["latents_tensor_sha256"],
            f"{tensor} vs {W4_EXPECTED['latents_tensor_sha256']}")
        add("w4_mp4", mp4 == W4_EXPECTED["mp4_sha256"], f"{mp4} vs {W4_EXPECTED['mp4_sha256']}")
    return checks


def run_launch(launch: Launch, *, record_dir: Path, adapter: na.NeuronAdapter,
               iters: int, records: Mapping[str, Any]) -> dict[str, Any]:
    out = run_dir(record_dir, launch)
    root = cache_root(record_dir, launch)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cores = wait_cores_free()
    if launch.fresh_cache:
        shutil.rmtree(root, ignore_errors=True)
        if root.exists() and any(root.rglob("*")):
            raise RuntimeError(f"{root} is not empty after rm -rf")
    root.mkdir(parents=True, exist_ok=True)
    before = host_state(root)
    cfg = config_for(launch.guidance)
    record: dict[str, Any] = {"launch": dataclasses.asdict(launch), "cache_root": str(root),
                              "cores_free": cores, "state_before": before,
                              "provenance": provenance()}
    if launch.drop_page_cache:
        record["page_cache_drop"] = drop_page_cache()
        record["state_after_drop"] = host_state(root)
    cold = records.get(f"{gkey(launch.guidance)}/script-cold")
    reference = None
    if cold and cold.get("summary"):
        reference = cold["summary"]["iterations"][0]["latents_sha256"]
    if launch.kind == "bench":
        result = adapter.run_bench(cfg, out, cache_root=str(root), expect=launch.expect,
                                   iters=iters, timeout=BENCH_TIMEOUT_S[launch.expect])
        after = host_state(root)
        added = new_neffs(before["neff"], after["neff"])
        record.update(run=result["launch"], summary=result["summary"], state_after=after,
                      new_neffs=added)
        record["checks"] = judge_bench(launch, result["summary"], result["launch"], added,
                                       None if launch.name == "script-cold" else reference)
    else:
        command = None
        if launch.kind == "cli-w4":
            command = w5_command(adapter.python, output=out / "out.mp4", cache_dir=root,
                                 work_dir=out / "work")
        result = adapter.run_cli_generate(cfg, out, cache_root=str(root), command=command,
                                          timeout=CLI_TIMEOUT_S)
        after = host_state(root)
        added = new_neffs(before["neff"], after["neff"])
        record.update(run=result["launch"], cli=result, state_after=after, new_neffs=added)
        record["checks"] = judge_cli(launch, result, added,
                                     after["hlo_files"] - before["hlo_files"], reference)
    record["ok"] = all(c["ok"] for c in record["checks"] if c["hard"])
    (out / "record.json").write_text(json.dumps(record, indent=2))
    return record


# ------------------------------------------------------------------------------- assembly


def _f(x: float | None, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _ms(x: float | None, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else (
        f"{x * 1000:.{digits}f}")


def _nanmax(values: Sequence[float | None]) -> float:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return max(values) if values else math.nan


def _st(samples: Sequence[float]) -> dict[str, Any]:
    return Stats.from_samples([s for s in samples if s is not None]).__dict__


def _ok_launches(records: Mapping[str, Any], g: float, names: Sequence[str]) -> list[dict]:
    out = []
    for name in names:
        rec = records.get(f"{gkey(g)}/{name}")
        if rec is not None and rec.get("ok"):
            out.append(rec)
    return out


def cli_timeline(rec: Mapping[str, Any] | None) -> dict[str, float] | None:
    """A CLI launch split by its files' mtimes: process start -> latents.pt saved by rank 0
    (torchrun spawn, load, generate) and latents.pt -> mp4 written (rank teardown, then the
    driver's host VAE load + decode + export), and mp4 -> process exit."""
    if not rec or not rec.get("cli"):
        return None
    latents, mp4 = Path(rec["cli"].get("latents", "")), Path(rec["cli"].get("mp4", ""))
    if not latents.is_file() or not mp4.is_file():
        return None
    run = rec["run"]
    t_latents, t_mp4 = latents.stat().st_mtime, mp4.stat().st_mtime
    return {"to_latents_s": t_latents - run["start_epoch"], "decode_s": t_mp4 - t_latents,
            "after_mp4_s": run["end_epoch"] - t_mp4}


def _cli_rank_max(rec: Mapping[str, Any], key: str) -> float | None:
    ranks = (rec.get("cli") or {}).get("parsed", {}).get("ranks", {})
    values = [r[key] for r in ranks.values()]
    return max(values) if values else None


def _counter_text(summary: Mapping[str, Any]) -> str:
    return " ".join(f"r{r['rank']} {r['counters']['run']['TotalCompilations']}/"
                    f"{r['counters']['run']['PersistentHits']}/"
                    f"{r['counters']['run']['InMemoryHits']}" for r in summary["ranks"])


def _range_text(stats: Mapping[str, Any], digits: int = 1, scale: float = 1.0) -> str:
    if not stats or not stats.get("n"):
        return "—"
    return (f"{stats['mean'] * scale:.{digits}f} (median {stats['median'] * scale:.{digits}f}; "
            f"{stats['min'] * scale:.{digits}f}-{stats['max'] * scale:.{digits}f}, n={stats['n']})")


def guidance_numbers(records: Mapping[str, Any], g: float) -> dict[str, Any] | None:
    """Everything the table and the BenchResult need for one guidance value, or None when
    the cold or the first warm script launch is missing."""
    cold = _ok_launches(records, g, ["script-cold"])
    warms = _ok_launches(records, g, ["script-warm1", "script-warm2", "script-warm3"])
    if not cold or not warms:
        return None
    cs, ws = cold[0]["summary"], [w["summary"] for w in warms]
    w1 = ws[0]
    it1 = w1["iterations"][0]
    resident_its = [it for s in ws for it in s["iterations"][1:]]
    cli_warm = _ok_launches(records, g, ["cli-warm1", "cli-warm2", "cli-warm3"])
    cli_cold = _ok_launches(records, g, ["cli-cold"])
    true_cold = records.get(f"{gkey(g)}/cli-truecold")
    w4 = records.get(f"{gkey(g)}/cli-w4check")
    cold_neffs = cold[0]["new_neffs"]
    compiling = max(cs["ranks"], key=lambda r: r["compile"]["compile_time_s"])
    return {
        "g": g, "cold": cs, "warms": ws, "w1": w1, "it1": it1, "resident_its": resident_its,
        "cold_record": cold[0], "warm_records": warms,
        "cli_warm": cli_warm, "cli_cold": cli_cold[0] if cli_cold else None,
        "true_cold": true_cold if true_cold and true_cold.get("ok") else None,
        "true_cold_record": true_cold, "w4": w4,
        "compile_wall_s": cs["rank_max"]["warmup_forward_s"] - w1["rank_max"]["warmup_forward_s"],
        "compiling_rank": compiling,
        "cold_new_neffs": len(cold_neffs),
        "cold_block_files": sum(1 for size in cold_neffs.values() if size >= BLOCK_NEFF_MIN_BYTES),
        "e2e_warm": _st([s["e2e_script_s"] for s in ws]),
        "e2e_latents_warm": _st([s["e2e_latents_s"] for s in ws]),
        "resident_e2e": _st([it["t_mp4_saved_s"] - it["t_start_s"] for it in resident_its]),
        "resident_latents": _st([it["t_latents_s"] - it["t_start_s"] for it in resident_its]),
        "resident_step": _st([d for it in resident_its for d in it["step"]["samples"]]),
        "resident_step_rank_max": _st([d for it in resident_its
                                       for d in it["step_rank_max"]["samples"]]),
        "resident_forward": _st([d for it in resident_its for d in it["forward"]["samples"]]),
        "cli_warm_wall": _st([r["run"]["wall_s"] for r in cli_warm]),
        "load_warm": _st([s["rank_max"]["load_checkpoint_s"] for s in ws]),
        "prepare_warm": _st([s["rank_max"]["prepare_runtime_s"] for s in ws]),
        "load_total_warm": _st([s["rank_max"]["load_s"] for s in ws]),
        "text_encoder_load_warm": _st([s["text_encoder_load_s"] for s in ws]),
        "launch_wall_warm": _st([r["run"]["wall_s"] for r in warms]),
        "spawn_warm": _st([s["prepare_start_epoch"] - r["run"]["start_epoch"]
                           for s, r in zip(ws, warms)]),
        "peak_gib": max(s["rank_max"]["peak_device_mem_gib"] for s in [cs, *ws]),
    }


def _checks_of(records: Mapping[str, Any], g: float) -> list[tuple[str, dict]]:
    out = []
    for key, rec in records.items():
        if key.startswith(gkey(g) + "/"):
            out += [(key.split("/", 1)[1], c) for c in rec.get("checks", [])]
    return out


def _step_row(it: Mapping[str, Any]) -> str:
    st = it["step"]
    return (f"{_ms(st['mean'])} / {_ms(st['p90'])} / {_ms(st['min'])} / {_ms(st['max'])}")


def w6_rows(n: Mapping[str, Any]) -> list[list[str]]:
    """The brief's table (W6 step 5) plus the rows errata P1/P2 added, for one guidance."""
    g, cs, ws, w1, it1 = n["g"], n["cold"], n["warms"], n["w1"], n["it1"]
    cit1 = cs["iterations"][0]
    res = n["resident_its"]
    comp = n["compiling_rank"]["compile"]
    rows: list[list[str]] = []
    rows.append([
        "compile wall (s)", "rank-max `warmup forward` cold − warm (first warm launch)",
        f"**{_f(n['compile_wall_s'])}** (= {_f(cs['rank_max']['warmup_forward_s'])} − "
        f"{_f(w1['rank_max']['warmup_forward_s'])})", "n/a", "n/a", TRN2["compile"]])
    blocks = sum(r["compile"]["block_compiled"] for r in cs["ranks"])
    eager = sum(r["compile"]["eager_compiled"] for r in cs["ranks"])
    rows.append([
        "NEFF compile (s) / NEFFs", "compiler-only `CompilationTime` sum on the compiling rank "
        "(block + eager-op; a sum over NEFFs compiled concurrently, not a wall clock); Dynamo "
        "frame compile (trace + lowering); new `.neff` files in the row's cache",
        f"{_f(comp['compile_time_s'])} (rank {n['compiling_rank']['rank']}: block "
        f"{_f(comp['block_compile_s'])} + eager-op {_f(comp['eager_compile_s'])}; Dynamo "
        f"{_f(comp['dynamo_compile_inner_s'])}, lowering {_f(comp['dynamo_lower_s'], 2)}) / "
        f"{n['cold_new_neffs']} new: {blocks} block (one per rank, {n['cold_block_files']} files "
        f">= 1 MiB) + {eager} eager-op",
        "0.0 / 0 new (" + ", ".join(
            f"w{i + 1}: {_f(max(r['compile']['compile_time_s'] for r in s['ranks']))} s, "
            f"{len(rec['new_neffs'])} new" for i, (s, rec) in
            enumerate(zip(ws, n["warm_records"]))) + ")",
        "0 compilations in iterations 2-3 (" + str(all(it["no_compile"] for it in res)) + ")",
        "transformer 255 s (AoT build)"])
    rows.append([
        "counters per rank", "whole process: TotalCompilations / PersistentHits / InMemoryHits",
        _counter_text(cs),
        "<br>".join(f"w{i + 1}: {_counter_text(s)}" for i, s in enumerate(ws)),
        "w1 it2-3 generate deltas: " + "; ".join(
            " ".join(f"r{k} {c['TotalCompilations']}/{c['PersistentHits']}/{c['InMemoryHits']}"
                     for k, c in enumerate(it["counters_by_rank"]))
            for it in w1["iterations"][1:]), ""])
    rows.append([
        "load (s)", "rank-max `load checkpoint` (each rank's bf16 slices of the fp32 shards)",
        _f(cs["rank_max"]["load_checkpoint_s"]), _range_text(n["load_warm"]), "n/a",
        TRN2["load"]])
    rows.append([
        "prepare_runtime / load() total (s)", "rank-max: runtime + process group bring-up; the "
        "whole `load()` (runtime init, build on meta, load checkpoint, per-block compile wrap, "
        "warm-up forward, text encoder load)",
        f"{_f(cs['rank_max']['prepare_runtime_s'])} / {_f(cs['rank_max']['load_s'])}",
        f"{_range_text(n['prepare_warm'])} / {_range_text(n['load_total_warm'])}", "n/a", ""])
    rows.append([
        "text encoder load (s)", "rank 0 `text encoder load` phase (umT5 fp32 "
        "`from_pretrained`, lazily mmap'd: the read lands in the first encode)",
        _f(cs["text_encoder_load_s"]), _range_text(n["text_encoder_load_warm"]), "n/a",
        TRN2["text_encoder_load"]])
    rows.append([
        "e2e (s), script", "rank 0: `prepare_runtime` start → first mp4 saved (load, warm-up, "
        "encode, 50 steps, host VAE decode at 12 threads, mp4)",
        f"**{_f(cs['e2e_script_s'])}**", f"**{_range_text(n['e2e_warm'])}**",
        f"generate only: {_range_text(n['resident_e2e'])}", TRN2["e2e"]])
    rows.append([
        "e2e (s), script, latent output", "`prepare_runtime` start → final latents (no decode)",
        _f(cs["e2e_latents_s"]), _range_text(n["e2e_latents_warm"]),
        f"generate only: {_range_text(n['resident_latents'])}", ""])
    rows.append([
        "launch wall (s), script", "driver: torchrun spawn → exit (3 generates + decodes); "
        "spawn + imports → `prepare_runtime`",
        f"{_f(n['cold_record']['run']['wall_s'])}; spawn {_f(cs['prepare_start_epoch'] - n['cold_record']['run']['start_epoch'])}",
        f"{_range_text(n['launch_wall_warm'])}; spawn {_range_text(n['spawn_warm'])}", "n/a", ""])
    cc = n["cli_cold"]
    rows.append([
        "e2e (s), CLI", "`difflet generate --backend neuron` process wall (torchrun ranks, "
        "then the driver's host VAE decode at OMP_NUM_THREADS=3)",
        (f"{_f(cc['run']['wall_s'])} (own fresh cache; {len(cc['new_neffs'])} new .neff)"
         if cc else "—"),
        _range_text(n["cli_warm_wall"]), "n/a", TRN2["e2e"]])
    tc = n["true_cold"]
    tcr = n["true_cold_record"]
    if tc:
        drop = tc.get("page_cache_drop") or {}
        tc_text = (f"{_f(tc['run']['wall_s'])} (Cached {_f(drop.get('before', {}).get('Cached_gib'))}"
                   f" → {_f(drop.get('after', {}).get('Cached_gib'))} GiB at the drop; NEFF warm)")
    elif tcr is None:
        tc_text = "skipped (no launch recorded)"
    else:
        tc_text = "failed: " + "; ".join(c["name"] for c in tcr.get("checks", []) if not c["ok"])
    rows.append(["e2e (s), CLI, true cold", "page cache dropped (`sudo -n`), NEFF cache warm: "
                 "the trn2 definition of a cold start", tc_text, "n/a", "n/a",
                 "394 (page cache dropped)"])
    cli_lw = [_cli_rank_max(r, "load_warmup_s") for r in n["cli_warm"]]
    cli_st = [_cli_rank_max(r, "stage_s") for r in n["cli_warm"]]
    rows.append([
        "CLI rank load+warm-up / stage (s)", "the `[wan]` rank lines, rank-max (load+warm-up "
        "includes `prepare_runtime`)",
        (f"{_f(_cli_rank_max(cc, 'load_warmup_s'))} / {_f(_cli_rank_max(cc, 'stage_s'))}"
         if cc else "—"),
        " ; ".join(f"{_f(a)} / {_f(b)}" for a, b in zip(cli_lw, cli_st)), "n/a", ""])

    def timeline(rec):
        t = cli_timeline(rec)
        return "—" if t is None else f"{_f(t['to_latents_s'])} / {_f(t['decode_s'])}"

    rows.append([
        "CLI start → latents / → mp4 (s)", "file mtimes: process start → latents.pt saved by "
        "rank 0; latents.pt → mp4 written (rank teardown + the driver's host VAE load, decode "
        "at 3 threads, export)", timeline(cc),
        " ; ".join(timeline(r) for r in n["cli_warm"]),
        f"true cold: {timeline(n['true_cold'])}" if n["true_cold"] else "n/a", ""])

    def stages(it):
        neg = f" (+ negative {_f(it['negative_encode_s'], 2)})" if it["negative_encode_s"] else ""
        return (f"{_f(it['text_encode_s'], 2)}{neg} / {_f(it['denoise_s'])} / "
                f"{_f(it['host_decode_s'])} (VAE {_f(it['vae_decode_s'])})")

    rows.append([
        "text encode / denoise / VAE decode (s)", "rank-0 stage clocks: prompt encode (the "
        "negative prompt is encoded lazily inside the first denoise iteration), `_denoise`, "
        "host decode incl. VAE load and mp4 write", stages(cit1), stages(it1),
        stages(w1["iterations"][1]) if len(w1["iterations"]) > 1 else "—", ""])
    rows.append([
        "DiT step median (ms)", "inter-step deltas of the real loop, `torch.neuron.synchronize` "
        "both sides, step 0 excluded, n=49, rank 0",
        f"evidence only: {_ms(cit1['step']['median'])}",
        f"**{_ms(it1['step']['median'])}** (w2 {_ms(ws[1]['iterations'][0]['step']['median']) if len(ws) > 1 else '—'}, "
        f"w3 {_ms(ws[2]['iterations'][0]['step']['median']) if len(ws) > 2 else '—'})",
        f"{_ms(n['resident_step']['median'])} (pooled, n={n['resident_step']['n']})",
        TRN2["step"]])
    rows.append([
        "DiT step rank-max median (ms)", "per step, the max over the 4 ranks (what the request "
        "waits for)", _ms(cit1["step_rank_max"]["median"]),
        _ms(it1["step_rank_max"]["median"]), _ms(n["resident_step_rank_max"]["median"]), ""])
    rs = n["resident_step"]
    rows.append([
        "DiT step mean / p90 / min / max (ms)", "same series", _step_row(cit1), _step_row(it1),
        f"{_ms(rs['mean'])} / {_ms(rs['p90'])} / {_ms(rs['min'])} / {_ms(rs['max'])}", ""])
    nf = 100 if g > 1.0 else 50
    rf = n["resident_forward"]
    rows.append([
        "DiT forward (ms)", f"per forward (host→device inputs, forward, sync, copy back), "
        f"n={nf}; median (mean / p90)",
        f"{_ms(cit1['forward']['median'])}",
        f"{_ms(it1['forward']['median'])} ({_ms(it1['forward']['mean'])} / "
        f"{_ms(it1['forward']['p90'])})",
        f"{_ms(rf['median'])} ({_ms(rf['mean'])} / {_ms(rf['p90'])}, n={rf['n']})", ""])
    rows.append([
        "throughput (steps/s)", "1 / mean step",
        f"evidence only: {_f(1.0 / cit1['step']['mean'], 3)}", _f(1.0 / it1["step"]["mean"], 3),
        _f(1.0 / rs["mean"], 3) if rs.get("mean") else "—", TRN2["throughput"]])
    resident_peak = [max(it["max_memory_allocated_gib_by_rank"][k] for it in res)
                     for k in range(len(w1["ranks"]))] if res else []
    rows.append([
        "peak device mem (GiB/rank)", "`torch_neuronx.max_memory_allocated()` per rank (the "
        "process's peak so far: after the launch, after generates 2-3)",
        " ".join(_f(r["peak_device_mem_gib"], 2) for r in cs["ranks"]),
        " ".join(_f(r["peak_device_mem_gib"], 2) for r in w1["ranks"]),
        " ".join(_f(v, 2) for v in resident_peak) or "—", TRN2["peak"]])

    def fb(summary):
        its = summary["iterations"]
        load = all(r["load_fallbacks"] == [] for r in summary["ranks"])
        gen = all(it["checks"]["fallbacks"] for it in its)
        unw = all(it["checks"]["unwarmed_shapes"] for it in its)
        return f"{'[]' if load and gen else 'NOT EMPTY'} / {'[]' if unw else 'NOT EMPTY'}"

    rows.append(["fallbacks / unwarmed shapes", "`track_fallbacks()` (load and every generate) "
                 "/ `app.unwarmed_shapes`, every rank", fb(cs),
                 " ; ".join(fb(s) for s in ws),
                 "[] / []" if all(it["checks"]["fallbacks"] and it["checks"]["unwarmed_shapes"]
                                  for it in res) else "NOT EMPTY", ""])
    spread_max = _nanmax([s["resident_step_median_spread"] for s in ws])
    rows.append(["throttle", "sysfs `ham_throttle` `entries`/`max` per core before vs right "
                 "after every generate (semantics undocumented: a flag, not a verdict); step "
                 "series p90/median (<= 1.10) and first-10 vs last-10 medians (<= 5 %)",
                 throttle_text([cs]), throttle_text(ws),
                 f"step medians of the 3 generates per launch: spread {_f(spread_max * 100, 2)}% "
                 "(max over the warm launches; flag above 2 %)", ""])
    def frames(it):
        fr = it.get("frames") or {}
        return (f"{fr.get('shape')} finite={fr.get('finite')} [{_f(fr.get('min'), 3)}, "
                f"{_f(fr.get('max'), 3)}] mean {_f(fr.get('mean'), 3)} std "
                f"{_f(fr.get('std'), 3)}; {it.get('mp4_bytes')} B mp4 "
                f"{str(it.get('mp4_sha256'))[:16]}…")

    rows.append(["output", "decoded frames (first generate of the launch; resident: generate "
                 "2 of the first warm launch): shape, finite, min/max/mean/std; mp4 bytes, "
                 "sha256", frames(cit1), frames(it1),
                 frames(w1["iterations"][1]) if len(w1["iterations"]) > 1 else "—",
                 TRN2["output"]])

    def distinct(summaries, select):
        values = sorted({select(it) for s in summaries for it in s["iterations"]
                         if isinstance(select(it), str)})
        return f"{len(values)} distinct: " + ", ".join(v[:16] + "…" for v in values)

    rows.append(["final latents", "sha256 of the fp32 latents, equal on the 4 ranks of every "
                 "generate (else the launch failed)",
                 distinct([cs], lambda it: it["latents_sha256"]),
                 distinct(ws, lambda it: it["latents_sha256"] if it["index"] == 1 else None),
                 distinct(ws, lambda it: it["latents_sha256"] if it["index"] > 1 else None),
                 "—"])
    w4 = n["w4"]
    if w4 is not None:
        failed = [c["name"] for c in w4.get("checks", []) if not c["ok"]]
        insp = (w4.get("cli") or {}).get("inspect", {})
        rows.append(["W4 reproduction (D80)", "one extra warm CLI run with W4's parameters: "
                     "latents and mp4 sha256 equal to W4's", "", (
                         "yes" if w4.get("ok") else f"NO ({failed})")
                     + f": latents {str(insp.get('tensor_sha256'))[:16]}…, file "
                     f"{str(insp.get('file_sha256'))[:16]}…, mp4 "
                     f"{str((insp.get('mp4') or {}).get('sha256'))[:16]}…; wall "
                     f"{_f(w4['run']['wall_s'])} s", "", ""])
    # no blank cell: "—" where a column does not apply (the brief's template leaves those blank)
    return [[cell if str(cell).strip() else "—" for cell in row] for row in rows]


def throttle_text(summaries: Sequence[Mapping[str, Any]]) -> str:
    """sysfs ``entries``/``max`` before vs right after each generate (a flag only), the max
    values seen, and the statistical checks of the step series."""
    changed, total, step_fail, maxima = 0, 0, [], set()
    for s in summaries:
        for it in s["iterations"]:
            total += 1
            changed += bool(it.get("throttle_changed"))
            for key in ("throttle_before", "throttle_after"):
                for fields in na.throttle_fields(it.get(key)).values():
                    maxima.add(fields.get("max", "?"))
            if not it["step_checks"]["ok"]:
                step_fail.append(it["index"])
    tails = [it["step_checks"]["tail_ratio"] for s in summaries for it in s["iterations"]]
    drifts = [it["step_checks"]["drift"] for s in summaries for it in s["iterations"]]
    return (f"sysfs entries/max changed across {changed}/{total} generates (max seen "
            f"{sorted(maxima)}); p90/median max {_f(_nanmax(tails), 3)}, first-10 vs last-10 "
            f"max {_f(_nanmax(drifts) * 100, 2)}%"
            + (f"; FLAGGED iterations {step_fail}" if step_fail else "; within limits"))


def bench_result(n: Mapping[str, Any], *, records: Mapping[str, Any], device: str,
                 toolchain: Mapping[str, str], prov: Mapping[str, Any]) -> dict[str, Any]:
    g, cs, w1, it1 = n["g"], n["cold"], n["w1"], n["it1"]
    cfg = config_for(g)
    comp = n["compiling_rank"]["compile"]
    fr = it1.get("frames") or {}
    res = BenchResult(
        model_id=cfg.model_id, model_type=cfg.model_type, backend="neuron", device=device,
        dtype=cfg.dtype, parallel=cfg.parallel_dict(),
        shape={"height": cfg.height, "width": cfg.width, "num_frames": cfg.num_frames},
        steps=cfg.steps, toolchain=dict(toolchain), config_label=cfg.config_label,
        # the end of this guidance's last launch: a re-render reproduces the file
        timestamp=datetime.datetime.fromtimestamp(
            max(rec["run"]["end_epoch"] for key, rec in records.items()
                if key.startswith(gkey(g) + "/")),
            datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    res.compile_seconds = n["compile_wall_s"]
    res.compile_breakdown = {
        "warmup_forward_cold_s": cs["rank_max"]["warmup_forward_s"],
        "warmup_forward_warm_s": w1["rank_max"]["warmup_forward_s"],
        "neff_compile_s": comp["compile_time_s"],
        "block_neff_compile_s": comp["block_compile_s"],
        "eager_op_neff_compile_s": comp["eager_compile_s"],
        "dynamo_frame_compile_cold_s": comp["dynamo_compile_inner_s"],
        "dynamo_frame_compile_warm_s": w1["ranks"][0]["compile"]["dynamo_compile_inner_s"],
        "dynamo_lower_s": comp["dynamo_lower_s"],
        "load_and_warmup_cold_s": cs["rank_max"]["load_s"],
        "load_and_warmup_warm_s": w1["rank_max"]["load_s"],
    }
    res.load_seconds = cs["rank_max"]["load_checkpoint_s"]
    res.e2e_cold_seconds = cs["e2e_script_s"]
    res.e2e_warm = n["e2e_warm"]
    res.step_latency = it1["step"]
    res.step_basis = (f"real loop, neuron synchronize both sides, step 0 excluded, "
                      f"n={it1['step']['n']}")
    res.stage_seconds = {
        "text_encoder_load_s": w1["text_encoder_load_s"] or 0.0,
        "text_encode": it1["text_encode_s"] or 0.0,
        "negative_encode (inside denoise)": it1["negative_encode_s"] or 0.0,
        "denoise": it1["denoise_s"] or 0.0,
        f"vae_decode (host, {it1['decode_threads']} threads)": it1["host_decode_s"] or 0.0,
    }
    res.throughput = {"DiT steps/s": 1.0 / it1["step"]["mean"]}
    res.peak_device_mem_gb = n["peak_gib"]
    res.output = OutputInfo(
        shape=fr.get("shape"), dtype=fr.get("dtype"), finite=fr.get("finite"),
        min=fr.get("min"), max=fr.get("max"), mean=fr.get("mean"), std=fr.get("std"),
        note=f"decoded frames of the first warm generate; {it1.get('mp4_bytes')} byte mp4 "
             f"sha256 {it1.get('mp4_sha256')}; latents (1,16,3,60,104) sha256 "
             f"{it1['latents_sha256']}").__dict__
    res.notes = guidance_notes(n, records=records, prov=prov)
    status = "ok"
    if any(not c["ok"] and c["hard"] for _, c in _checks_of(records, g)):
        status = "partial"
    res.status = status
    d = res.to_dict()
    d.update({"config_slug": SLUG, "device_slug": "trn2", "revision": cfg.revision,
              "seed": cfg.seed, "prompt": cfg.prompt, "guidance_scale": g,
              "output_kind": cfg.output_kind})
    d["neuron"] = {
        "table": w6_rows(n),
        "e2e_latents_warm": n["e2e_latents_warm"],
        "e2e_cli_warm": n["cli_warm_wall"],
        "e2e_cli_cold_s": n["cli_cold"]["run"]["wall_s"] if n["cli_cold"] else None,
        "e2e_cli_true_cold_s": n["true_cold"]["run"]["wall_s"] if n["true_cold"] else None,
        "resident_e2e": n["resident_e2e"], "resident_step": n["resident_step"],
        "step_rank_max": it1["step_rank_max"], "forward": it1["forward"],
        "checks": [{"launch": name, **c} for name, c in _checks_of(records, g)],
        "provenance": prov,
    }
    return d


def guidance_notes(n: Mapping[str, Any], *, records: Mapping[str, Any],
                   prov: Mapping[str, Any]) -> list[str]:
    g, cs, ws = n["g"], n["cold"], n["warms"]
    r0 = cs["ranks"][0]
    notes = [
        f"Commit {prov['head']} in {prov['worktree']} for all {prov['launches']} launches; "
        + ("no change under difflet/ at any launch"
           if prov["difflet_diff_vs_head_at_runs"] == [""] else
           f"difflet/ changes at some launch: {prov['difflet_diff_vs_head_at_runs']}")
        + "; uncommitted files at the launches: "
        + ", ".join(sorted({line.split(maxsplit=1)[-1] for s in prov["status_at_runs"]
                            for line in s.splitlines() if line.strip()}))
        + " (per-launch sha256 of the benchmark files in neuron.provenance). guidance_scale="
        f"{g:g}: {'two batch-1 DiT forwards per step (CFG)' if g > 1 else 'one forward per step'}.",
        "compile_seconds is the rank-max `warmup forward` of the cold script launch minus the "
        "first warm launch's (Dynamo trace + StableHLO lowering + neuronx-cc + NEFF load that "
        "the cache removes); nothing is built ahead of time, so there is no AoT compile phase.",
        "e2e_cold_seconds / e2e_warm are the script's `prepare_runtime` -> mp4 saved on rank 0 "
        "(NEFF cache empty / populated; the OS page cache was hot for both, Cached "
        f"{_f(n['cold_record']['state_before']['meminfo'].get('Cached_gib'))} GiB before the cold "
        "launch). The trn2 row's cold start dropped the page cache: see the true-cold CLI row.",
        f"Counters (Total/PersistentHits/InMemoryHits, whole process): cold {_counter_text(cs)}; "
        + "; ".join(f"warm{i + 1} {_counter_text(s)}" for i, s in enumerate(ws)),
        f"Cold NEFFs: {n['cold_new_neffs']} new .neff files ({n['cold_block_files']} >= 1 MiB "
        "block NEFFs); per rank compiled block/eager-op: " + ", ".join(
            f"r{r['rank']} {r['compile']['block_compiled']}/{r['compile']['eager_compiled']}"
            for r in cs["ranks"]) + "; dynamo unique graphs per rank " + str(
            [r["dynamo_unique_graphs"] for r in cs["ranks"]]),
        f"Threads: ranks OMP_NUM_THREADS={r0['env'].get('OMP_NUM_THREADS')} (torch default "
        f"{r0['threads']['default']}); umT5 encode pinned at {r0['threads']['encoder']} (D70); "
        f"the script's host VAE decode on rank 0 at {r0['threads']['decode']}; the CLI driver's "
        "host decode at its torch default, 3 under OMP_NUM_THREADS=3 (pre-flight: 97.4 s at 3 "
        "vs ~65 s at 6, same mp4).",
        "Peak device memory is torch_neuronx.max_memory_allocated() per rank, in GiB (2**30).",
    ]
    bench_hashes = prov["files_sha256_at_runs"].get("benchmark/neuron_wan_bench.py", [])
    notes.append(
        f"The device-side script benchmark/neuron_wan_bench.py had {len(bench_hashes)} sha256 "
        f"value(s) across the {prov['launches']} launches"
        + (" (byte-identical)." if len(bench_hashes) == 1 else f": {bench_hashes}."))
    rejudged = sorted(key for key, rec in records.items() if rec.get("rejudged"))
    if rejudged:
        notes.append(
            "Re-judged after the run (the stored verdict was a driver false positive; the record "
            "as it ran is kept as record.original.json): " + ", ".join(
                f"{key} (stored failed {records[key]['rejudged']['stored_failed']})"
                for key in rejudged)
            + ". The job stopped there and resumed with --skip-existing; no launch was "
            "repeated.")
    clean = all(c["ok"] for _, c in _checks_of(records, g)
                if c["name"] in ("fallbacks", "unwarmed_shapes", "compile_clean"))
    notes.append(
        ("Fallbacks and unwarmed shapes were [] on every rank of every launch and generate"
         if clean else "Fallbacks or unwarmed shapes were NOT empty somewhere (see checks)")
        + " (track_fallbacks only sees eager ops; the compiled blocks are proven by the "
        "counters).")
    flagged = [f"{name}:{c['name']}" for name, c in _checks_of(records, g)
               if not c["ok"] and not c["hard"]]
    if flagged:
        notes.append("FLAGGED (statistical/evidence checks, not enforced): " + ", ".join(flagged))
    failed = [f"{name}:{c['name']}" for name, c in _checks_of(records, g)
              if not c["ok"] and c["hard"]]
    if failed:
        notes.append("FAILED (hard checks): " + ", ".join(failed))
    return notes


# ------------------------------------------------------------------------------- markdown


def demote_headings(text: str) -> str:
    """Every Markdown heading one level down (``#`` -> ``##``), code blocks untouched."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("#"):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def drop_section(text: str, heading: str) -> str:
    """``text`` without the section that starts at the line ``heading`` (to the next heading of
    the same or a higher level, or the end)."""
    level = len(heading) - len(heading.lstrip("#"))
    out, skipping, fenced = [], False, False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and line.strip() == heading:
            skipping = True
            continue
        elif skipping and not fenced and line.startswith("#"):
            if len(line) - len(line.lstrip("#")) <= level:
                skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def table_md(rows: Sequence[Sequence[str]]) -> str:
    head = ["metric", "definition", "cold launch", "warm launch (n=3)",
            "resident (2nd/3rd generate)", "trn2 AoT baseline (20 steps, g=1.0)"]
    lines = ["| " + " | ".join(head) + " |", "|---|---|---:|---:|---:|---:|"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def check_lines(checks: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per launch: the passing check names, every flag and failure with its detail,
    and the counter / file rule's detail (the warm proof) even when it passes."""
    by_launch: dict[str, list[Mapping[str, Any]]] = {}
    for c in checks:
        by_launch.setdefault(c["launch"], []).append(c)
    lines = []
    for launch, items in by_launch.items():
        passed = [c["name"] for c in items if c["ok"]]
        text = f"- `{launch}`: {len(passed)}/{len(items)} pass"
        rule = [c for c in items if c["name"].startswith(("counter_rule", "file_rule"))]
        for c in rule:
            text += f"; {c['name']}: {c['detail']}"
        for c in items:
            if not c["ok"]:
                text += f"; **{'FAIL' if c['hard'] else 'flag'}** {c['name']}: {c['detail']}"
        lines.append(text)
    return lines


def render_markdown(results: Mapping[str, Mapping[str, Any]], *, plan_note: str) -> str:
    order = [k for k in (f"{g:.1f}" for g in (HEADLINE_GUIDANCE, 1.0)) if k in results]
    first = results[order[0]]
    prov = first["neuron"]["provenance"]
    L = [f"# Benchmark — Wan2.1 T2V 14B on the neuron backend (TorchNeuron, TP4, "
         f"{first['steps']} steps)", "",
         f"**Status:** {', '.join(f'g={k}: ' + results[k]['status'] for k in order)}  ",
         f"**Backend:** neuron (torch-neuronx, per-block `torch.compile`)  ",
         f"**Device:** {first['device']}  ",
         "**Timestamp (last launch):** "
         + ", ".join(f"g={k}: {results[k]['timestamp']}" for k in order) + "  ",
         f"**Commit:** `{prov['head']}` in `{prov['worktree']}`", "",
         f"> Configuration: {first['config_label']}; 480x832x9 (latent 1x16x3x60x104, 4,680 "
         "tokens, 512 text tokens), bf16, seed 42, prompt \"" + first["prompt"] + "\".", "",
         "## Headline", "",
         "| guidance | compile (s) | e2e cold (s) | e2e warm (s, n=3 mean) | DiT step median (ms) "
         "| DiT forward median (ms) | peak HBM (GiB/rank) |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k in order:
        r = results[k]
        L.append(f"| {k} | {_f(r['compile_seconds'])} | {_f(r['e2e_cold_seconds'])} | "
                 f"{_f(r['e2e_warm']['mean'])} | {_ms(r['step_latency']['median'])} | "
                 f"{_ms(r['neuron']['forward']['median'])} | {_f(r['peak_device_mem_gb'], 2)} |")
    L += ["", "e2e = the benchmark script's `prepare_runtime` → mp4 saved on rank 0 (NEFF cache "
          "empty for cold, populated for warm; OS page cache hot in both). The user-facing "
          "`difflet generate` walls, the page-cache-cold row and the resident (serving) rows are "
          "in the tables below. DiT step = one denoise iteration of the real loop (at g=5.0 two "
          "forwards + the host UniPC step).", "", plan_note, ""]
    for k in order:
        r = results[k]
        g = float(k)
        L += [f"## Guidance {k}" + (" (headline: real video, CFG = two batch-1 forwards per step)"
                                     if g == HEADLINE_GUIDANCE else
                                     " (one forward per step; the trn2-comparable row)"), "",
              table_md(r["neuron"]["table"]), "",
              "Checks per launch (every check's detail is in the JSON, `neuron.checks`):", ""]
        L += check_lines(r["neuron"]["checks"]) + [""]
    L += ["## Definitions", "",
          "- **compile time**: rank-max `app.phase_seconds[\"warmup forward\"]` on the launch with "
          "an empty cache (`DIFFLET_COMPILE_CACHE` fresh: `neuron/{neff,neff_local,hlo}` empty) "
          "minus the same on the next launch with the populated cache; the compiler-only "
          "`CompilationCache.CompilationTime` sum and the new `.neff` count beside it.",
          "- **e2e cold / warm**: the bench script's rank-0 wall from `prepare_runtime` to the "
          "first mp4 saved; warm proven per launch by per-rank `TotalCompilations == 0 and "
          "PersistentHits >= 1` and no new `.neff`. The CLI rows are the `difflet generate` "
          "process wall (torchrun spawn and imports included; host VAE in the driver); the CLI "
          "prints no counters, so a warm CLI launch is proven by its cache gaining no file.",
          "- **resident**: generates 2 and 3 in the same process (`InMemoryHits`), the serving "
          "steady state; e2e from the generate's start to its mp4.",
          "- **DiT step**: `RealLoopStepTimer(sync=torch.neuron.synchronize)`, one stamp per "
          "denoise iteration (after the scheduler step), the device synchronized before the loop, "
          "step 0 excluded, n=49; median is the headline. The trn2 554.8 ms is an isolated "
          "forward timer at 20 steps, g=1.0 (not re-measured with this rule).", "",
          "## Reproduction (neuron)", "", "```bash",
          "# one supervised job, guidance 5.0 then 1.0 (about 1.5-2 h of device time):",
          "HF_HUB_OFFLINE=1 PYTHONSAFEPATH=1 PYTHONPATH=<worktree> \\",
          "    <venv>/bin/python -m benchmark.neuron_cold_warm --record-dir <records>",
          "# re-render this file from the recorded launches (no device):",
          "PYTHONSAFEPATH=1 PYTHONPATH=<worktree> <venv>/bin/python -m benchmark.neuron_cold_warm \\",
          "    --record-dir <records> --assemble-only",
          "# the user-facing command each CLI row timed:",
          "difflet generate --backend neuron --exec-mode compile --host-vae \\",
          "    --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision "
          "38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \\",
          "    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 \\",
          "    --steps 50 --guidance-scale 5.0 --seed 42 \\",
          "    --prompt \"a cinematic shot of a red fox running through a snowy forest\" "
          "--output out.mp4", "```", ""]
    for k in order:
        body = drop_section(report.render(results[k]), "## Reproduction")
        L += [f"## Appendix: generic report, guidance {k} (`benchmark/report.py`)", "",
              "Rendered by the shared renderer, headings demoted, its Trainium reproduction "
              "section dropped. Its fixed labels assume the AoT protocol: \"compile (AOT, "
              "one-time)\" is the warm-up compile above, and \"cold start (page cache dropped)\" "
              "is the NEFF-cold launch with a hot page cache (the true-cold CLI row is the "
              "page-cache-cold one), so its sentence that the cold/warm gap is the cold disk "
              "read does not apply: here the gap is the block compile and the cold eager-op "
              "NEFFs.", "", demote_headings(body), ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------------- main


def load_records(record_dir: Path, *, rejudge: bool = False) -> dict[str, Any]:
    """The recorded launches by ``g<guidance>/<name>``. With ``rejudge`` every bench launch is
    summarized again from its ``result-rank<r>.json`` and every launch judged again, so a
    re-render reflects this file's logic rather than the one that ran."""
    order = [l.name for l in build_plan([HEADLINE_GUIDANCE])]

    def plan_order(path: Path) -> tuple[str, int]:
        name = path.parent.name
        return (path.parent.parent.name, order.index(name) if name in order else len(order))

    records = {}
    for path in sorted((record_dir / "runs").glob("g*/*/record.json"), key=plan_order):
        records[f"{path.parent.parent.name}/{path.parent.name}"] = json.loads(path.read_text())
    if not rejudge:
        return records
    # script-cold first: the others compare their latents with it
    for key in sorted(records, key=lambda k: not k.endswith("/script-cold")):
        rec = records[key]
        launch = Launch(**rec["launch"])
        out = run_dir(record_dir, launch)
        cold = records.get(f"{gkey(launch.guidance)}/script-cold") or {}
        reference = None
        if cold.get("summary") and launch.name != "script-cold":
            reference = cold["summary"]["iterations"][0]["latents_sha256"]
        if launch.kind == "bench":
            if rec["run"]["exit_code"] == 0:
                results = na.read_rank_results(out)
                rec["summary"] = na.summarize_bench_launch(
                    results, steps=int(results[0]["args"]["steps"]),
                    guidance_scale=float(results[0]["args"]["guidance_scale"]))
            rec["checks"] = judge_bench(launch, rec.get("summary"), rec["run"],
                                        rec["new_neffs"], reference)
        else:
            log = Path(rec["run"].get("log") or out / "cli.log")
            if log.is_file():
                parsed = na.parse_cli_log(log.read_text(errors="replace"))
                rec["cli"]["parsed"], rec["cli"]["checks"] = parsed, na.cli_checks(parsed)
            hlo_delta = rec["state_after"]["hlo_files"] - rec["state_before"]["hlo_files"]
            rec["checks"] = judge_cli(launch, rec["cli"], rec["new_neffs"], hlo_delta,
                                      reference)
        rec["ok"] = all(c["ok"] for c in rec["checks"] if c["hard"])
    return records


def run_provenance(records: Mapping[str, Any]) -> dict[str, Any]:
    """What the launches ran, from their records (not from the moment of assembly, so a
    re-render reproduces the files): the HEADs, the worktree, the git status, neuronx-cc and
    every benchmark file's sha256 values across the launches (one value = unchanged)."""
    provs = [rec["provenance"] for rec in records.values() if rec.get("provenance")]
    heads = sorted({p["head"] for p in provs})
    files: dict[str, list[str]] = {}
    for p in provs:
        for rel, digest in p["files_sha256"].items():
            if digest not in files.setdefault(rel, []):
                files[rel].append(digest)
    return {
        "head": heads[0] if len(heads) == 1 else heads,
        "worktree": sorted({p["worktree"] for p in provs})[0] if provs else None,
        "status_at_runs": sorted({p["status"] for p in provs}),
        "difflet_diff_vs_head_at_runs": sorted({p["difflet_diff_vs_head"] for p in provs}),
        "neuronx_cc": sorted({line for p in provs for line in p["neuronx_cc"]}),
        "neuron_cc_flags": sorted({str(p["neuron_cc_flags"]) for p in provs}),
        "files_sha256_at_runs": files,
        "launches": len(provs),
    }


def assemble(records: Mapping[str, Any], *, adapter: na.NeuronAdapter, out_json: Path,
             out_md: Path, plan_note: str) -> dict[str, Any]:
    device, toolchain = adapter.device_info(), adapter.toolchain()
    prov = run_provenance(records)
    results = {}
    for g in (HEADLINE_GUIDANCE, 1.0):
        n = guidance_numbers(records, g)
        if n is not None:
            results[f"{g:.1f}"] = bench_result(n, records=records, device=device,
                                               toolchain=toolchain, prov=prov)
    if not results:
        raise RuntimeError("no guidance value has both a cold and a warm script launch")
    walls = {k: r["compile_seconds"] for k, r in results.items()}
    if len(walls) == 2:
        a, b = walls.values()
        spread = abs(a - b) / min(a, b)
        note = (f"Cold compile walls: g=5.0 {_f(walls.get('5.0'))} s, g=1.0 "
                f"{_f(walls.get('1.0'))} s (same block graph and shape; one cold compile per "
                f"guidance value, each on its own fresh cache): they differ by "
                f"{spread * 100:.1f}%" + (", more than 15 %: reported separately, not averaged."
                                          if spread > 0.15 else "."))
        plan_note = plan_note + " " + note
    top_key = "1.0" if "1.0" in results else next(iter(results))
    doc = dict(results[top_key])  # the MATRIX entry's guidance (1.0): wan_2_1.json's schema
    doc["results_by_guidance"] = results
    doc["headline_guidance"] = HEADLINE_GUIDANCE
    doc["plan_note"] = plan_note
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2))
    out_md.write_text(render_markdown(results, plan_note=plan_note))
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--record-dir", type=Path, required=True)
    p.add_argument("--guidance", type=float, nargs="+", default=list(GUIDANCES))
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warm-launches", type=int, default=3)
    p.add_argument("--cli-warm", type=int, default=3)
    p.add_argument("--true-cold", choices=("auto", "yes", "no"), default="auto")
    p.add_argument("--no-w4-check", action="store_true",
                   help="skip the W4-parameter CLI run (smoke runs only)")
    p.add_argument("--steps", type=int, default=None,
                   help="override the MATRIX entry's 50 steps (smoke runs only; the result "
                        "files then describe a smoke, not the benchmark)")
    p.add_argument("--out-json", type=Path,
                   default=na.REPO_ROOT / "benchmark" / "trn2" / f"{SLUG}.json")
    p.add_argument("--out-md", type=Path,
                   default=na.REPO_ROOT / "benchmark" / "trn2" / f"{SLUG}.md")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--assemble-only", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="reuse launches whose record.json passed (resume after a stop)")
    args = p.parse_args(argv)
    record_dir = args.record_dir.absolute()
    global STEPS_OVERRIDE
    STEPS_OVERRIDE = args.steps

    true_cold = {"yes": True, "no": False}.get(args.true_cold)
    sudo_ok = sudo_available()
    if true_cold is None:
        true_cold = sudo_ok
    plan = build_plan(args.guidance, warm_launches=args.warm_launches, cli_warm=args.cli_warm,
                      true_cold=true_cold,
                      w4_guidance=None if args.no_w4_check else HEADLINE_GUIDANCE)
    plan_note = (f"Run order: guidance {', '.join(f'{g:g}' for g in args.guidance)}; per "
                 f"guidance {', '.join(l.name for l in plan if l.guidance == args.guidance[0])}."
                 + ("" if true_cold else " The true-cold row was skipped: `sudo -n true` "
                    f"{'works but --true-cold no was given' if sudo_ok else 'failed'}."))
    adapter = na.NeuronAdapter(cache_dir=str(record_dir / "caches"),
                               log_dir=str(record_dir / "runs"), iters=args.iters)
    if args.assemble_only:
        doc = assemble(load_records(record_dir, rejudge=True), adapter=adapter,
                       out_json=args.out_json, out_md=args.out_md, plan_note=plan_note)
        print(f"[neuron_cold_warm] wrote {args.out_json} and {args.out_md} "
              f"({list(doc['results_by_guidance'])})", flush=True)
        return 0

    leaked = [name for name in na.CACHE_ENV_VARS if os.environ.get(name)]
    print(f"[neuron_cold_warm] shell cache variables (dropped from every launch env): "
          f"{leaked or 'none'}; sudo -n: {sudo_ok}; plan {len(plan)} launches", flush=True)
    cfg0 = config_for(args.guidance[0])
    for launch in plan:
        root = cache_root(record_dir, launch)
        out = run_dir(record_dir, launch)
        if launch.kind == "bench":
            cmd = na.bench_command(adapter.python, model_dir="<snapshot>", out_dir=out,
                                   steps=cfg0.steps, guidance_scale=launch.guidance,
                                   iters=args.iters, prompt=cfg0.prompt, seed=cfg0.seed,
                                   height=cfg0.height, width=cfg0.width,
                                   num_frames=cfg0.num_frames, expect=launch.expect)
        elif launch.kind == "cli-w4":
            cmd = w5_command(adapter.python, output=out / "out.mp4", cache_dir=root,
                             work_dir=out / "work")
        else:
            cmd = na.cli_generate_command(adapter.python, config_for(launch.guidance),
                                          output=out / "out.mp4", cache_dir=root,
                                          work_dir=out / "work")
        flags = ("fresh cache, " if launch.fresh_cache else "") + (
            "page cache dropped, " if launch.drop_page_cache else "")
        print(f"[plan] {gkey(launch.guidance)}/{launch.name}: {launch.kind}, {flags}expect "
              f"{launch.expect}, DIFFLET_COMPILE_CACHE={root}\n        {' '.join(cmd)}",
              flush=True)
    if args.dry_run:
        env = na.launch_env(os.environ, cache_root=record_dir / "caches" / "example",
                            export_cache_dirs=True, metrics_dir=record_dir / "example")
        print("[dry-run] bench launch env: " + json.dumps(
            {k: env.get(k) for k in (*na.LAUNCH_ENV, "PYTHONPATH", "DIFFLET_COMPILE_CACHE",
                                     *na.CACHE_ENV_VARS, "NKI_ENABLE_TRACE_CACHE",
                                     "TORCH_NEURONX_ENABLED_METRIC_TABLES",
                                     "TORCH_NEURONX_METRICS_DIR")}, indent=1), flush=True)
        return 0

    records = load_records(record_dir, rejudge=True) if args.skip_existing else {}
    for key, rec in records.items():
        # Persist a verdict that changed on rejudging; the record as it ran is kept beside it.
        path = record_dir / "runs" / key / "record.json"
        stored = json.loads(path.read_text())
        if stored["ok"] != rec["ok"] or stored["checks"] != rec["checks"]:
            original = path.with_name("record.original.json")
            if not original.exists():
                original.write_text(path.read_text())
            rec["rejudged"] = {"epoch": time.time(), "stored_ok": stored["ok"],
                               "stored_failed": [c["name"] for c in stored["checks"]
                                                 if not c["ok"]]}
            path.write_text(json.dumps(rec, indent=2))
            print(f"[rejudged] {key}: ok {stored['ok']} -> {rec['ok']}", flush=True)
    stopped = None
    for launch in plan:
        key = f"{gkey(launch.guidance)}/{launch.name}"
        if args.skip_existing and records.get(key, {}).get("ok"):
            print(f"[skip] {key}: recorded and ok", flush=True)
            continue
        print(f"[launch] {key} starting at {datetime.datetime.now():%H:%M:%S}", flush=True)
        rec = run_launch(launch, record_dir=record_dir, adapter=adapter, iters=args.iters,
                         records=records)
        records[key] = rec
        bad = [c for c in rec["checks"] if not c["ok"]]
        print(f"[launch] {key}: {'ok' if rec['ok'] else 'FAILED'} wall "
              f"{rec['run']['wall_s']:.1f}s exit {rec['run']['exit_code']}; "
              + ("; ".join(f"{'FAIL' if c['hard'] else 'flag'} {c['name']}: {c['detail']}"
                           for c in bad) or "all checks pass"), flush=True)
        if not rec["ok"]:
            stopped = key
            break
    try:
        doc = assemble(records, adapter=adapter, out_json=args.out_json, out_md=args.out_md,
                       plan_note=plan_note)
        print(f"[neuron_cold_warm] wrote {args.out_json} and {args.out_md} "
              f"({list(doc['results_by_guidance'])})", flush=True)
    except Exception as exc:  # noqa: BLE001 - the launch records stay for --assemble-only
        print(f"[neuron_cold_warm] assembly failed: {type(exc).__name__}: {exc}", flush=True)
        if stopped is None:
            raise
    if stopped is not None:
        print(f"[neuron_cold_warm] STOPPED after {stopped} (hard check failed)", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
