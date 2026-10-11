"""Wan2.1 T2V 14B on the neuron backend, TP4, 480x832x9, 50 steps: compile time, e2e cold,
e2e warm and DiT step time, per guidance value. The driver; it never touches the device.

Per guidance value (5.0 first, the headline; then 1.0, the trn2-comparable row) it runs, one
device job at a time, each launch after the cores-free check:

1. ``script-cold``: a fresh ``DIFFLET_COMPILE_CACHE`` (empty ``neuron/{neff,neff_local,hlo}``)
   and one torchrun launch of ``benchmark/neuron_wan_bench.py`` with three generates (the
   first is the cold number, the others the resident rows);
2. ``script-warm1..3``: the same launch three more times on that cache (warm: per-rank
   ``TotalCompilations == 0 and PersistentHits >= 1``, no new ``.neff``);
3. ``cli-cold``: the user-facing ``difflet generate --backend neuron`` on its own fresh cache
   (``g<g>-cli-default``; the user-facing e2e cold; its block compile is a second, intentional
   compile sample, not the compile-time number);
4. at guidance 5.0 only, ``cli-w4check``: one warm ``difflet generate`` with W4's exact
   parameters (the W5 section-4 command), which must reproduce W4's latents and mp4 (D80); a
   mismatch stops the job;
5. ``cli-warm1..3``: ``difflet generate`` again on the CLI cache;
6. ``cli-truecold`` (when ``sudo -n`` works): the page cache dropped first, NEFF cache warm,
   the trn2 definition of a cold start.

The bench launches export ``OMP_NUM_THREADS=3`` for the ranks (brief 8.5); the CLI launches
export none, so torchrun and the ``difflet generate`` driver keep their defaults (ruling D82).

Every launch records the host before and after (``free -g``, ``/proc/meminfo``, NEFF / HLO /
``/tmp`` cache file lists, the sysfs throttle strings, neuronx-cc version, thread counts, git
HEAD, the difflet/ tree and the benchmark files' sha256) under
``<record-dir>/runs/g<guidance>/<launch>/``. Hard failures (a non-zero exit, a fallback, an
unwarmed shape, ranks or iterations with different latents, a broken counter rule on a warm
launch, the W4 hashes, an exception) stop the job; statistical checks (step tail and drift,
the 2 % resident rule) are flagged, not enforced (a flag sets the guidance's status to
"flagged"). A completed job assembles one ``BenchResult`` per guidance value into
``benchmark/trn2/wan_2_1_neuron.json`` and ``.md`` (guidance 5.0 first); a stopped job and
``--assemble-only`` write ``<record-dir>/results/`` unless ``--out-json``/``--out-md`` say
otherwise. Stop a running job with ``kill -INT -<pgid>`` (the job script's process group):
the driver stops the running launch's ranks before it exits.

    PYTHONSAFEPATH=1 PYTHONPATH=<worktree> <venv>/bin/python -m benchmark.neuron_cold_warm \\
        --record-dir <records>

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
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.adapters import neuron as na
from benchmark.harness import BenchResult, OutputInfo, Stats
from benchmark.models import MATRIX
from benchmark.neuron_report import HEADLINE_GUIDANCE, _counter_text, _f, render_markdown, w6_rows

SLUG = "wan_2_1_neuron"
GUIDANCES = (5.0, 1.0)
BENCH_FILES = ("benchmark/neuron_cold_warm.py", "benchmark/neuron_wan_bench.py",
               "benchmark/adapters/neuron.py", "benchmark/neuron_report.py",
               "benchmark/harness.py", "benchmark/report.py", "benchmark/models.py",
               "tests/numerical/neuron_parity.py")
#: W4's 50-step final run, reproduced by W5's CLI cold/warm runs ($REC/W5/SUMMARY.txt).
W4_EXPECTED = {
    "prompt": "a cinematic shot of a red fox running through a snowy forest",
    "seed": 42, "steps": 50, "guidance_scale": 5.0,
    "latents_file_sha256": "e639f3146c3619406267c317c9798b8ead7300535c7835a7d7c7c0de54769028",
    "latents_tensor_sha256":
        "8e5eaffed4cf6f3a79a94dc11664c4f6b49767375fc6f5c451a39330bed990f6",
    "mp4_sha256": "a9f9004fc1343eab167e70f4522f382243f40b7b83765b726e01d02cb01b7f34",
}
BLOCK_NEFF_MIN_BYTES = 1 << 20
BENCH_TIMEOUT_S = {"cold": 3600.0, "warm": 2400.0}
CLI_TIMEOUT_S = 2400.0


# ------------------------------------------------------------------------------- the plan


@dataclasses.dataclass(frozen=True)
class Launch:
    name: str
    kind: str              # "bench" | "cli" | "cli-w4"
    guidance: float
    cache: str             # which cache root of this guidance: "main" | "cli-default"
    expect: str            # "cold" | "warm"
    fresh_cache: bool = False
    drop_page_cache: bool = False


#: The CLI rows' own cache (D82: a CLI cold fills it, the warm CLI launches read it).
CLI_CACHE = "cli-default"


def build_plan(guidances: Sequence[float] = GUIDANCES, *, warm_launches: int = 3,
               cli_warm: int = 3, true_cold: bool = True,
               w4_guidance: float | None = HEADLINE_GUIDANCE) -> list[Launch]:
    plan: list[Launch] = []
    for g in guidances:
        plan.append(Launch("script-cold", "bench", g, "main", "cold", fresh_cache=True))
        plan += [Launch(f"script-warm{i}", "bench", g, "main", "warm")
                 for i in range(1, warm_launches + 1)]
        plan.append(Launch("cli-cold", "cli", g, CLI_CACHE, "cold", fresh_cache=True))
        if w4_guidance is not None and g == w4_guidance:
            plan.append(Launch("cli-w4check", "cli-w4", g, CLI_CACHE, "warm"))
        plan += [Launch(f"cli-warm{i}", "cli", g, CLI_CACHE, "warm")
                 for i in range(1, cli_warm + 1)]
        if true_cold:
            plan.append(Launch("cli-truecold", "cli", g, CLI_CACHE, "warm",
                               drop_page_cache=True))
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


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


#: The driver-side code this process imported, hashed once at import: the files on disk may
#: change while a job runs, the code that runs does not.
LOADED_SHA256 = {
    rel: _sha256(na.REPO_ROOT / rel)
    for rel in ("benchmark/neuron_cold_warm.py", "benchmark/adapters/neuron.py",
                "benchmark/neuron_report.py")
}


def provenance() -> dict[str, Any]:
    """What one launch runs: HEAD, the difflet/ tree (its git tree id plus any uncommitted
    diff), the benchmark files on disk (the bench script is read per launch), the driver code
    loaded at import, neuronx-cc and NEURON_CC_FLAGS."""
    def git(*args: str) -> str:
        return _run_text(["git", "-C", str(na.REPO_ROOT), *args])

    return {
        "head": git("rev-parse", "HEAD"),
        "worktree": str(na.REPO_ROOT),
        "status": git("status", "--porcelain"),
        "difflet_tree": git("rev-parse", "HEAD:difflet"),
        "difflet_diff_vs_head": git("diff", "--stat", "HEAD", "--", "difflet"),
        "files_sha256": {rel: _sha256(na.REPO_ROOT / rel) for rel in BENCH_FILES},
        "loaded_sha256": dict(LOADED_SHA256),
        "neuronx_cc": _run_text([str(Path(sys.executable).parent / "neuronx-cc"), "--version"]
                                ).splitlines()[0:1],
        "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS"),
        "python": sys.executable,
    }


def host_info(adapter: na.NeuronAdapter) -> dict[str, Any]:
    """The host facts a re-render needs, recorded once per job (M-2): the device string, the
    package versions and whether ``sudo -n`` works (the true-cold rows)."""
    return {"device": adapter.device_info(), "toolchain": adapter.toolchain(),
            "sudo_available": sudo_available(), "epoch": time.time()}


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


def set_aside(out: Path) -> Path | None:
    """Keep a previous attempt's directory as ``<name>.failed-<epoch>`` (``load_records``
    skips those) instead of deleting its evidence."""
    if not out.exists():
        return None
    target = out.with_name(f"{out.name}.failed-{int(time.time())}")
    out.rename(target)
    return target


def run_launch(launch: Launch, *, record_dir: Path, adapter: na.NeuronAdapter,
               iters: int, records: Mapping[str, Any],
               host: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = run_dir(record_dir, launch)
    root = cache_root(record_dir, launch)
    previous = set_aside(out)
    out.mkdir(parents=True)
    cfg = config_for(launch.guidance)
    record: dict[str, Any] = {"launch": dataclasses.asdict(launch),
                              "cache_root": os.path.relpath(root, record_dir),
                              "host": dict(host or {}),
                              "previous_attempt": previous and previous.name}
    cold = records.get(f"{gkey(launch.guidance)}/script-cold")
    reference = None
    if cold and cold.get("summary"):
        reference = cold["summary"]["iterations"][0]["latents_sha256"]
    try:
        # A failure before the launch (cores busy, a cache that will not empty) is a failed
        # record like any other: the job stops there and still assembles results/.
        record["cores_free"] = wait_cores_free()
        if launch.fresh_cache:
            shutil.rmtree(root, ignore_errors=True)
            if root.exists() and any(root.rglob("*")):
                raise RuntimeError(f"{root} is not empty after rm -rf")
        root.mkdir(parents=True, exist_ok=True)
        before = host_state(root)
        record.update(state_before=before, provenance=provenance())
        if launch.drop_page_cache:
            record["page_cache_drop"] = drop_page_cache()
            record["state_after_drop"] = host_state(root)
        if launch.kind == "bench":
            result = adapter.run_bench(cfg, out, cache_root=str(root), expect=launch.expect,
                                       iters=iters, timeout=BENCH_TIMEOUT_S[launch.expect])
            after = host_state(root)
            added = new_neffs(before["neff"], after["neff"])
            record.update(run=result["launch"], summary=result["summary"], state_after=after,
                          new_neffs=added)
            record["checks"] = judge_bench(launch, result["summary"], result["launch"], added,
                                           None if launch.name == "script-cold" else reference)
            if result.get("error"):
                record["checks"].append({"name": "rank_results", "ok": False, "hard": True,
                                         "detail": result["error"]})
        else:
            command = None
            if launch.kind == "cli-w4":
                command = w5_command(adapter.python, output=out / "out.mp4", cache_dir=root,
                                     work_dir=out / "work")
            result = adapter.run_cli_generate(cfg, out, cache_root=str(root), command=command,
                                              timeout=CLI_TIMEOUT_S)
            after = host_state(root)
            added = new_neffs(before["neff"], after["neff"])
            record.update(run=result["launch"], cli=result, state_after=after, new_neffs=added,
                          timeline=result.get("timeline"), threads=result.get("threads"))
            record["checks"] = judge_cli(launch, result, added,
                                         after["hlo_files"] - before["hlo_files"], reference)
    except Exception as exc:  # noqa: BLE001 - recorded, then the job stops
        record.setdefault("run", {"exit_code": None, "wall_s": None, "timed_out": False})
        record["checks"] = [{"name": "exception", "ok": False, "hard": True,
                             "detail": f"{type(exc).__name__}: {exc}"}]
    record["ok"] = all(c["ok"] for c in record["checks"] if c["hard"])
    (out / "record.json").write_text(json.dumps(record, indent=2))
    return record


# ------------------------------------------------------------------------------- assembly


def _st(samples: Sequence[float]) -> dict[str, Any]:
    return Stats.from_samples([s for s in samples if s is not None]).__dict__


def _ok_launches(records: Mapping[str, Any], g: float, names: Sequence[str]) -> list[dict]:
    out = []
    for name in names:
        rec = records.get(f"{gkey(g)}/{name}")
        if rec is not None and rec.get("ok"):
            out.append(rec)
    return out


WARM_NAMES = ("script-warm1", "script-warm2", "script-warm3")
CLI_WARM_NAMES = ("cli-warm1", "cli-warm2", "cli-warm3")


def guidance_numbers(records: Mapping[str, Any], g: float) -> dict[str, Any] | None:
    """Everything the table and the BenchResult need for one guidance value, or None when
    ``script-cold`` or ``script-warm1`` (the headline step series) is missing or failed. The
    warm launches keep their names, so a missing one never relabels another (M-9)."""
    cold = _ok_launches(records, g, ["script-cold"])
    if not cold or not _ok_launches(records, g, ["script-warm1"]):
        return None
    warm_names = [name for name in WARM_NAMES if _ok_launches(records, g, [name])]
    warms = [_ok_launches(records, g, [name])[0] for name in warm_names]
    cs, ws = cold[0]["summary"], [w["summary"] for w in warms]
    w1 = ws[0]
    it1 = w1["iterations"][0]
    resident_its = [it for s in ws for it in s["iterations"][1:]]
    cli_warm_names = [name for name in CLI_WARM_NAMES if _ok_launches(records, g, [name])]
    cli_warm = [_ok_launches(records, g, [name])[0] for name in cli_warm_names]
    cli_cold = _ok_launches(records, g, ["cli-cold"])
    true_cold = records.get(f"{gkey(g)}/cli-truecold")
    w4 = records.get(f"{gkey(g)}/cli-w4check")
    cold_neffs = cold[0]["new_neffs"]
    compiling = max(cs["ranks"], key=lambda r: r["compile"]["compile_time_s"])
    return {
        "g": g, "cold": cs, "warms": ws, "w1": w1, "it1": it1, "resident_its": resident_its,
        "warm_names": warm_names, "cli_warm_names": cli_warm_names,
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
        "threads": thread_facts(records, g),
    }


def thread_facts(records: Mapping[str, Any], g: float) -> dict[str, Any]:
    """The thread counts the rows ran with, from the records (never assumed): the bench
    launches' ``OMP_NUM_THREADS``; per CLI row kind the launch env's value and the driver's
    torch threads (its host VAE decode)."""
    def cli(names):
        recs = [records.get(f"{gkey(g)}/{name}") for name in names]
        threads = [(r.get("threads") or {}) for r in recs if r]
        return {"omp": sorted({str(t.get("launch_env_omp_num_threads") or "unset")
                               for t in threads}),
                "driver": sorted({t.get("driver_torch_threads") for t in threads
                                  if t.get("driver_torch_threads") is not None})}

    bench = [records.get(f"{gkey(g)}/{name}") for name in ("script-cold", *WARM_NAMES)]
    return {
        "bench_omp": sorted({str(((r.get("run") or {}).get("env") or {}).get("OMP_NUM_THREADS"))
                             for r in bench if r}),
        "cli": cli(("cli-cold", "cli-w4check", *CLI_WARM_NAMES)),
        "true_cold": cli(("cli-truecold",)),
    }


def _checks_of(records: Mapping[str, Any], g: float) -> list[tuple[str, dict]]:
    out = []
    for key, rec in records.items():
        if key.startswith(gkey(g) + "/"):
            out += [(key.split("/", 1)[1], c) for c in rec.get("checks", [])]
    return out


def status_of(records: Mapping[str, Any], g: float) -> str:
    """"partial" when a hard check failed, "flagged" when only a statistical / evidence check
    did (M-1: shown, never fatal), else "ok"."""
    checks = [c for _, c in _checks_of(records, g)]
    if any(not c["ok"] and c["hard"] for c in checks):
        return "partial"
    if any(not c["ok"] for c in checks):
        return "flagged"
    return "ok"


def bench_result(n: Mapping[str, Any], *, records: Mapping[str, Any], device: str,
                 toolchain: Mapping[str, str], prov: Mapping[str, Any],
                 extra_notes: Sequence[str] = ()) -> dict[str, Any]:
    g, cs, w1, it1 = n["g"], n["cold"], n["w1"], n["it1"]
    cfg = config_for(g)
    comp = n["compiling_rank"]["compile"]
    fr = it1.get("frames") or {}
    ends = [rec["run"]["end_epoch"] for key, rec in records.items()
            if key.startswith(gkey(g) + "/") and (rec.get("run") or {}).get("end_epoch")]
    res = BenchResult(
        model_id=cfg.model_id, model_type=cfg.model_type, backend="neuron", device=device,
        dtype=cfg.dtype, parallel=cfg.parallel_dict(),
        shape={"height": cfg.height, "width": cfg.width, "num_frames": cfg.num_frames},
        steps=cfg.steps, toolchain=dict(toolchain), config_label=cfg.config_label,
        # the end of this guidance's last launch: a re-render reproduces the file
        timestamp=datetime.datetime.fromtimestamp(max(ends), datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"),
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
    res.notes = guidance_notes(n, records=records, prov=prov) + list(extra_notes)
    res.status = status_of(records, g)
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
        "threads": n["threads"],
        "checks": [{"launch": name, **c} for name, c in _checks_of(records, g)],
        "provenance": prov,
    }
    return d


def _kind(name: str) -> str:
    return "script" if name.startswith("script-") else (
        "true-cold CLI" if name == "cli-truecold" else "CLI")


def provenance_note(records: Mapping[str, Any], g: float, prov: Mapping[str, Any]) -> str:
    """Which HEAD each kind of row ran at, and whether difflet/ was the same tree with no
    uncommitted change at every launch (the NEFF keys depend on it, R1 finding 2)."""
    heads: dict[str, set[str]] = {}
    for key, rec in records.items():
        if key.startswith(gkey(g) + "/") and rec.get("provenance"):
            heads.setdefault(rec["provenance"]["head"][:7], set()).add(_kind(key.split("/")[1]))
    trees = prov.get("difflet_trees") or []
    same = len(trees) == 1 and prov["difflet_diff_vs_head_at_runs"] == [""]
    return ("HEAD at the launches: " + "; ".join(
        f"{head} ({', '.join(sorted(kinds))} rows)" for head, kinds in sorted(heads.items()))
        + f", all in {prov['worktree']}; difflet/ "
        + (f"identical at every launch (tree {trees[0][:12]}, no uncommitted change)" if same
           else f"NOT the same at every launch: trees {trees}, diffs "
                f"{prov['difflet_diff_vs_head_at_runs']}") + ".")


BENCH_SCRIPT = "benchmark/neuron_wan_bench.py"
#: Known differences between the device-side script a launch ran and a later copy, by
#: (sha256 that ran, sha256 of the later copy).
BENCH_SCRIPT_CHANGES = {
    ("adb718622beb1a63b7e836caa4bc1953dc158602e7376052c5d14411d587ae7a",
     "36751f0ed5cad9e22c2503da3e7622fe2136a04f8bbd58c1c878adbb1792e89b"):
        "only the W6 review's M-10 change: its encode_prompt and _denoise wrappers take "
        "(*args, **kwargs) and tag a generate's encodes by call order (first = prompt); for "
        "the orchestrator's keyword-only calls (the prompt, then the empty negative prompt) the "
        "behaviour and the recorded tags are the same",
}


def bench_script_note(prov: Mapping[str, Any], current: str | None = None) -> str:
    """How many versions of the device-side script the launches ran, and how the copy in this
    tree relates to them (``current``: its sha256, read from the tree when not given)."""
    ran = prov["files_sha256_at_runs"].get(BENCH_SCRIPT, [])
    current = current or _sha256(na.REPO_ROOT / BENCH_SCRIPT)
    note = (f"The device-side script {BENCH_SCRIPT} had {len(ran)} sha256 value(s) across the "
            f"{prov['launches']} launches" + (f" ({ran[0][:12]}…, byte-identical)" if len(ran) == 1
                                                else f": {[h[:12] for h in ran]}"))
    if ran == [current]:
        return note + "; the copy in this tree is the same file."
    if len(ran) == 1:
        what = BENCH_SCRIPT_CHANGES.get((ran[0], current), "not described here (see git log)")
        return note + f"; the copy in this tree ({str(current)[:12]}…) differs from it: {what}."
    return note + f"; the copy in this tree is {str(current)[:12]}…."


#: The fix round's (FR) two `difflet generate --backend neuron` launches after its jemalloc
#: commit, in the CLI's default environment (records: /home/ubuntu/plans/wan21/records/FR).
FIX_ROUND_LAUNCHES = (
    "The fix round's two launches after that commit ran without it (only HF_HUB_OFFLINE=1 and "
    "DIFFLET_NEURON_PG_TIMEOUT_S=3600 exported, no DIFFLET_COMPILE_CACHE, --cache-dir a fresh "
    "directory that the warm launch reused; W4's parameters, guidance 5): cold exit 0, 220.9 s, "
    "47 new .neff under the "
    "--cache-dir (the CLI rows' 47 keys); warm exit 0, 158.0 s, no new cache file; latents.pt "
    "e639f3146c36… and mp4 a9f9004fc134… both times, the same as the CLI rows' and W4's")


def jemalloc_note(records: Mapping[str, Any], g: float) -> str:
    """What DIFFLET_NO_JEMALLOC the CLI rows' launch envs recorded, why, and the fix round's
    launches without it."""
    values = sorted({
        str((((rec.get("cli") or {}).get("launch") or {}).get("env") or {})
            .get("DIFFLET_NO_JEMALLOC"))
        for key, rec in records.items()
        if key.startswith(gkey(g) + "/") and _kind(key.split("/")[1]) != "script"})
    return ("jemalloc: every CLI row's recorded launch env had DIFFLET_NO_JEMALLOC="
            + "/".join(values) + " (Reproduction). Before the fix round's commit \"fix(cli): "
            "do not preload jemalloc for the torchrun (neuron) backend\" a plain `difflet "
            "generate --backend neuron` re-exec'd itself under the jemalloc LD_PRELOAD, which "
            "its torchrun ranks and their neuronx-cc compiles inherited; on this host the "
            "variable changed nothing (torch_neuronx is a source build without "
            "lib/libjemalloc.so, so nothing was preloaded either way). " + FIX_ROUND_LAUNCHES
            + ".")


def guidance_notes(n: Mapping[str, Any], *, records: Mapping[str, Any],
                   prov: Mapping[str, Any]) -> list[str]:
    g, cs, ws = n["g"], n["cold"], n["warms"]
    r0 = cs["ranks"][0]
    th = n["threads"]
    notes = [
        provenance_note(records, g, prov) + " Uncommitted files at the launches: "
        + (", ".join(sorted({line.split(maxsplit=1)[-1] for s in prov["status_at_runs"]
                             for line in s.splitlines() if line.strip()})) or "none")
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
        + "; ".join(f"{name.removeprefix('script-')} {_counter_text(s)}"
                    for name, s in zip(n["warm_names"], ws)),
        f"Cold NEFFs: {n['cold_new_neffs']} new .neff files ({n['cold_block_files']} >= 1 MiB "
        "block NEFFs); per rank compiled block/eager-op: " + ", ".join(
            f"r{r['rank']} {r['compile']['block_compiled']}/{r['compile']['eager_compiled']}"
            for r in cs["ranks"]) + "; dynamo unique graphs per rank " + str(
            [r["dynamo_unique_graphs"] for r in cs["ranks"]]),
        f"Threads (as recorded): script ranks OMP_NUM_THREADS={'/'.join(th['bench_omp'])} "
        f"(torch default {r0['threads']['default']}); umT5 encode pinned at "
        f"{r0['threads']['encoder']} (D70); the script's host VAE decode on rank 0 at "
        f"{r0['threads']['decode']}; CLI warm/cold launches: OMP_NUM_THREADS in the launch env "
        f"{'/'.join(th['cli']['omp'])} (ruling D82: none, so torchrun gives its ranks 1 and the "
        f"`difflet generate` driver decodes the VAE at {'/'.join(map(str, th['cli']['driver']))} "
        "threads)"
        + (f"; the true-cold launch: OMP_NUM_THREADS {'/'.join(th['true_cold']['omp'])}, driver "
           f"decode at {'/'.join(map(str, th['true_cold']['driver']))} threads"
           if th["true_cold"]["omp"] else "") + ".",
        "Peak device memory is torch_neuronx.max_memory_allocated() per rank, in GiB (2**30).",
        f"neuronx-cc: {', '.join(prov['neuronx_cc'])}; NEURON_CC_FLAGS at the launches: "
        f"{', '.join(prov['neuron_cc_flags'])}.",
    ]
    notes.append(bench_script_note(prov))
    notes.append(jemalloc_note(records, g))
    rejudged =sorted(key for key, rec in records.items() if rec.get("rejudged"))
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


# ----------------------------------------------------------------------------------- main


def load_records(record_dir: Path, *, rejudge: bool = False) -> dict[str, Any]:
    """The recorded launches by ``g<guidance>/<name>`` (``runs/g*/<name>/record.json``; a
    previous attempt set aside as ``<name>.failed-<epoch>`` is skipped). With ``rejudge``
    every bench launch is summarized again from its ``result-rank<r>.json`` and every launch
    judged again (a CLI launch from its own ``cli.log``), all located next to the record, so a
    re-render reflects this file's logic rather than the one that ran."""
    order = [launch.name for launch in build_plan([HEADLINE_GUIDANCE])]

    def plan_order(path: Path) -> tuple[str, int]:
        name = path.parent.name
        return (path.parent.parent.name, order.index(name) if name in order else len(order))

    records, where = {}, {}
    for path in sorted((record_dir / "runs").glob("g*/*/record.json"), key=plan_order):
        if ".failed-" in path.parent.name:
            continue
        key = f"{path.parent.parent.name}/{path.parent.name}"
        records[key], where[key] = json.loads(path.read_text()), path.parent
    if not rejudge:
        return records
    # script-cold first: the others compare their latents with it
    for key in sorted(records, key=lambda k: not k.endswith("/script-cold")):
        rec, out = records[key], where[key]
        launch = Launch(**rec["launch"])
        cold = records.get(f"{gkey(launch.guidance)}/script-cold") or {}
        reference = None
        if cold.get("summary") and launch.name != "script-cold":
            reference = cold["summary"]["iterations"][0]["latents_sha256"]
        if any(c["name"] == "exception" for c in rec.get("checks", [])):
            continue  # nothing ran to judge again
        if launch.kind == "bench":
            if rec["run"]["exit_code"] == 0:
                results = na.read_rank_results(out)
                rec["summary"] = na.summarize_bench_launch(
                    results, steps=int(results[0]["args"]["steps"]),
                    guidance_scale=float(results[0]["args"]["guidance_scale"]))
            rec["checks"] = judge_bench(launch, rec.get("summary"), rec["run"],
                                        rec["new_neffs"], reference)
        else:
            log = out / "cli.log"
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
    re-render reproduces the files): the HEADs (always a list), the difflet/ trees, the
    worktree, the git status, neuronx-cc and every benchmark file's sha256 values across the
    launches on disk and as loaded by the driver (one value = unchanged)."""
    provs = [rec["provenance"] for rec in records.values() if rec.get("provenance")]

    def values(field: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in provs:
            for rel, digest in (p.get(field) or {}).items():
                if digest not in out.setdefault(rel, []):
                    out[rel].append(digest)
        return out

    return {
        "head": sorted({p["head"] for p in provs}),
        "worktree": sorted({p["worktree"] for p in provs})[0] if provs else None,
        "difflet_trees": sorted({p["difflet_tree"] for p in provs if p.get("difflet_tree")}),
        "status_at_runs": sorted({p["status"] for p in provs}),
        "difflet_diff_vs_head_at_runs": sorted({p["difflet_diff_vs_head"] for p in provs}),
        "neuronx_cc": sorted({line for p in provs for line in p["neuronx_cc"]}),
        "neuron_cc_flags": sorted({str(p["neuron_cc_flags"]) for p in provs}),
        "files_sha256_at_runs": values("files_sha256"),
        "loaded_sha256_at_runs": values("loaded_sha256"),
        "launches": len(provs),
    }


def recorded_host(records: Mapping[str, Any]) -> dict[str, Any] | None:
    """The host facts the first launch (in plan order) recorded, or None (M-2)."""
    return next((rec["host"] for rec in records.values() if (rec.get("host") or {}).get("device")),
                None)


def _hhmm(epoch: float | None) -> str:
    if epoch is None:
        return "?"
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime("%H:%M")


def plan_note_of(records: Mapping[str, Any]) -> str:
    """The launches in the order they ran (start time), per guidance value (blocks in the order
    they started), grouped into stretches with the same HEAD and cache, with UTC windows."""
    blocks: dict[str, list[tuple[float, str, Mapping[str, Any]]]] = {}
    for key, rec in records.items():
        g, name = key.split("/", 1)
        start = (rec.get("run") or {}).get("start_epoch")
        blocks.setdefault(g, []).append((math.inf if start is None else start, name, rec))
    parts = []
    for g in sorted(blocks, key=lambda g: (min(s for s, _, _ in blocks[g]), g)):
        stretches: list[list[Any]] = []  # [head, cache, names, first start, last end]
        for start, name, rec in sorted(blocks[g], key=lambda item: (item[0], item[1])):
            head = ((rec.get("provenance") or {}).get("head") or "?")[:7]
            cache = f"{g}-{rec['launch']['cache']}"
            end = (rec.get("run") or {}).get("end_epoch")
            if stretches and stretches[-1][:2] == [head, cache]:
                stretches[-1][2].append(name)
                stretches[-1][4] = end
            else:
                stretches.append([head, cache, [name], None if start == math.inf else start, end])
        parts.append(f"g={g[1:].removesuffix('.0')}: " + "; then ".join(
            f"{', '.join(names)} ({_hhmm(first)}-{_hhmm(last)} UTC, HEAD {head}, cache {cache})"
            for head, cache, names, first, last in stretches))
    note = "Run order (by start time): " + " | ".join(parts) + "."
    if not any(key.endswith("/cli-truecold") for key in records):
        host = recorded_host(records) or {}
        note += (" The true-cold row was not run (`sudo -n` recorded as "
                 f"{host.get('sudo_available')}).")
    return note


def superseded_note(record_dir: Path) -> list[str]:
    """Launch records the record dir keeps but this result does not use (each subdirectory of
    ``runs/`` that is not a guidance block, with its README's first line)."""
    notes = []
    for directory in sorted(p for p in (record_dir / "runs").glob("*")
                            if p.is_dir() and not p.name.startswith("g")):
        launches = sorted(f"{p.parent.parent.name}/{p.parent.name}"
                          for p in directory.glob("*/*/record.json"))
        readme = directory / "README.txt"
        first = readme.read_text().splitlines()[0].rstrip(".") if readme.is_file() else ""
        notes.append(f"Superseded records kept under runs/{directory.name}/ (not used here): "
                     f"{', '.join(launches)}" + (f" — {first}" if first else "") + ".")
    return notes


def assemble(records: Mapping[str, Any], *, adapter: na.NeuronAdapter, out_json: Path,
             out_md: Path, plan_note: str | None = None,
             extra_notes: Sequence[str] = ()) -> dict[str, Any]:
    host = recorded_host(records)
    if host is None:  # records from before M-2: read the host now and say so
        host = host_info(adapter)
        extra_notes = [*extra_notes, "Host facts (device, toolchain) read at assembly."]
    prov = run_provenance(records)
    plan_note = plan_note or plan_note_of(records)
    results = {}
    for g in (HEADLINE_GUIDANCE, 1.0):
        n = guidance_numbers(records, g)
        if n is not None:
            results[f"{g:.1f}"] = bench_result(n, records=records, device=host["device"],
                                               toolchain=host["toolchain"], prov=prov,
                                               extra_notes=extra_notes)
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


def _interrupt(signum, frame):  # noqa: ARG001 - signal handler signature
    raise KeyboardInterrupt(f"signal {signum}")


def output_paths(args: argparse.Namespace, record_dir: Path, *, final: bool) -> tuple[Path, Path]:
    """Where the result files go: the repo's ``benchmark/trn2`` only for a completed job;
    ``--assemble-only`` and a stopped job write ``<record-dir>/results/`` (M-3); explicit
    ``--out-json``/``--out-md`` always win."""
    default = (na.REPO_ROOT / "benchmark" / "trn2") if final else (record_dir / "results")
    return (args.out_json or default / f"{SLUG}.json", args.out_md or default / f"{SLUG}.md")


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
    p.add_argument("--out-json", type=Path, default=None,
                   help="default: benchmark/trn2/wan_2_1_neuron.json for a completed job, "
                        "<record-dir>/results/ for --assemble-only or a stopped job")
    p.add_argument("--out-md", type=Path, default=None, help="as --out-json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--assemble-only", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="reuse launches whose record.json passes when judged again")
    args = p.parse_args(argv)
    record_dir = args.record_dir.absolute()
    global STEPS_OVERRIDE
    STEPS_OVERRIDE = args.steps
    adapter = na.NeuronAdapter(cache_dir=str(record_dir / "caches"),
                               log_dir=str(record_dir / "runs"), iters=args.iters)
    if args.assemble_only:
        out_json, out_md = output_paths(args, record_dir, final=False)
        doc = assemble(load_records(record_dir, rejudge=True), adapter=adapter,
                       out_json=out_json, out_md=out_md,
                       extra_notes=superseded_note(record_dir))
        print(f"[neuron_cold_warm] wrote {out_json} and {out_md} "
              f"({list(doc['results_by_guidance'])})", flush=True)
        return 0

    true_cold = {"yes": True, "no": False}.get(args.true_cold)
    sudo_ok = sudo_available()
    if true_cold is None:
        true_cold = sudo_ok
    plan = build_plan(args.guidance, warm_launches=args.warm_launches, cli_warm=args.cli_warm,
                      true_cold=true_cold,
                      w4_guidance=None if args.no_w4_check else HEADLINE_GUIDANCE)
    leaked = [name for name in (*na.CACHE_ENV_VARS, "OMP_NUM_THREADS") if os.environ.get(name)]
    print(f"[neuron_cold_warm] shell variables dropped from every launch env: "
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
        envs = {"bench": na.launch_env(os.environ, cache_root=record_dir / "caches" / "example",
                                       export_cache_dirs=True, omp_threads=na.BENCH_OMP_THREADS,
                                       metrics_dir=record_dir / "example"),
                "cli": na.launch_env(os.environ, cache_root=record_dir / "caches" / "example",
                                     export_cache_dirs=False)}
        for kind, env in envs.items():
            print(f"[dry-run] {kind} launch env: " + json.dumps(
                {k: env.get(k) for k in (*na.RECORDED_ENV, "TORCH_NEURONX_ENABLED_METRIC_TABLES",
                                         "TORCH_NEURONX_METRICS_DIR")}, indent=1), flush=True)
        return 0

    signal.signal(signal.SIGTERM, _interrupt)  # stop like Ctrl-C: the launch's ranks first
    host = host_info(adapter)
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
    try:
        for launch in plan:
            key = f"{gkey(launch.guidance)}/{launch.name}"
            if args.skip_existing and records.get(key, {}).get("ok"):
                print(f"[skip] {key}: recorded and ok", flush=True)
                continue
            print(f"[launch] {key} starting at {datetime.datetime.now():%H:%M:%S}", flush=True)
            rec = run_launch(launch, record_dir=record_dir, adapter=adapter, iters=args.iters,
                             records=records, host=host)
            records[key] = rec
            bad = [c for c in rec["checks"] if not c["ok"]]
            wall = rec["run"].get("wall_s")
            print(f"[launch] {key}: {'ok' if rec['ok'] else 'FAILED'} wall "
                  f"{_f(wall)}s exit {rec['run'].get('exit_code')}; "
                  + ("; ".join(f"{'FAIL' if c['hard'] else 'flag'} {c['name']}: {c['detail']}"
                               for c in bad) or "all checks pass"), flush=True)
            if not rec["ok"]:
                stopped = key
                break
    except KeyboardInterrupt as exc:
        # run_logged stopped the running launch (its ranks included) before this arrives
        stopped = f"an interrupt ({exc or 'SIGINT'})"
    out_json, out_md = output_paths(args, record_dir, final=stopped is None)
    try:
        doc = assemble(records, adapter=adapter, out_json=out_json, out_md=out_md,
                       extra_notes=superseded_note(record_dir))
        print(f"[neuron_cold_warm] wrote {out_json} and {out_md} "
              f"({list(doc['results_by_guidance'])})", flush=True)
    except Exception as exc:  # noqa: BLE001 - the launch records stay for --assemble-only
        print(f"[neuron_cold_warm] assembly failed: {type(exc).__name__}: {exc}", flush=True)
        if stopped is None:
            raise
    if stopped is not None:
        print(f"[neuron_cold_warm] STOPPED after {stopped}", flush=True)
        return 130 if stopped.startswith("an interrupt") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
