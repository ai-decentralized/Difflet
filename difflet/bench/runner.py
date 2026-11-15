"""Run one recipe's metrics on a pinned lane and write its result (design §7, §9, §10).

A :class:`RecipeRun` owns one result record. Its steps are separate methods (``compile_metric``,
``prepare``, ``sample("e2e_cold")``, ``sample("e2e_warm")``, ``quality``, ``finish``) so a suite can
interleave several recipes' launches in ABBA order; ``run_recipe`` calls them in sequence.

Every measured launch is preceded by an idle check of the lane's devices and CPUs (the host is
shared), taken before anything else touches the host; a busy lane aborts the run instead of
producing a contaminated number. Whatever ends a run (an abort, an error, Ctrl-C, SIGTERM), the
result is still written, marked invalid.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from difflet.bench import host as H
from difflet.bench import launch as L
from difflet.bench import memory as M
from difflet.bench import pagecache, provenance, results
from difflet.bench import timing as T
from difflet.bench.recipe import (RECIPES_FILE, Plan, engine_parsers, file_sha256, fingerprint,
                                  split_flags)

METRICS = ("compile", "e2e_cold", "e2e_warm", "dit_step", "memory", "quality")
DEFAULT_METRICS = ("compile", "e2e_cold", "e2e_warm", "dit_step", "memory")
DEFAULT_REPEATS = {"e2e_cold": 2, "e2e_warm": 3}
#: Warm launches carry dit_step and memory; any of the three needs them.
WARM_CARRIED = frozenset({"e2e_warm", "dit_step", "memory"})
# shortcut: initial thresholds, calibrate on device data (design §17)
RESIDENT_MAX_AFTER_EVICT = 0.02
RESIDENT_MIN_AFTER_PREREAD = 0.98
STEP_SUM_TOLERANCE = 0.02
STEP_TAIL_RATIO = 1.10
STEP_DRIFT = 0.05
MEMORY_DISAGREEMENT = 0.25
LAUNCH_TIMEOUT_S = 4 * 3600.0


class RunAborted(RuntimeError):
    """The run cannot continue (busy lane, failed preparation); the partial result is written."""


@dataclass
class Lane:
    devices: list[int]
    cpus: list[int]

    @property
    def cores(self) -> str:
        return H.devices_to_cores(self.devices)

    @property
    def core_ids(self) -> set[int]:
        return set(H.parse_cpu_list(self.cores))


@dataclass
class RunConfig:
    metrics: list[str]
    lane: Lane
    repeats: int | None = None
    timeout_s: float = LAUNCH_TIMEOUT_S
    #: "cpu_fp32", "none" (dump only), a result JSON path, or (from a suite) a dump directory.
    quality_ref: str | None = None
    results_root: Path | None = None
    device_name: str | None = None


def device_slug() -> str:
    """``trn3.48xlarge`` -> ``trn3``: the results folder for this kind of host."""
    instance = H.topology()["instance_type"] or "unknown"
    match = re.match(r"(trn|inf)\d+", instance)
    return match.group(0) if match else instance


def output_suffix(plan: Plan) -> str:
    return ".mp4" if plan.recipe.args.get("num-frames") else ".png"


def _dest(flag: str) -> str | None:
    action = engine_parsers()["generate"]._option_string_actions.get(flag)
    return action.dest if action is not None else None


def effective_mismatches(plan: Plan, engine_args: dict[str, Any] | None) -> list[str]:
    """Flags bench asked for that the engine did not run with (from the driver's ``args`` event).

    Compared on the parsed values both sides hold (bench parses with the same parser), so a
    default the engine fills in or a type conversion is not a mismatch; only a flag that was set
    and then changed is."""
    if engine_args is None:
        return ["the engine reported no final arguments (no 'args' timing event)"]
    requested = vars(plan.generate_ns)
    out = []
    for key in {k for k, _ in split_flags([*plan.recipe.argv(), *plan.passthrough])}:
        dest = _dest(f"--{key}")
        if dest is None or dest not in requested:
            continue
        want, got = requested[dest], engine_args.get(dest)
        if str(want) != str(got):
            out.append(f"--{key}: requested {want!r}, the engine ran {got!r}")
    return sorted(out)


def recipe_files(cache: Path, model_id: str, revision: str) -> list[Path]:
    """The files a generate of this model at this revision reads (design §7, e2e_cold): the hub
    snapshot, and the per-revision artifacts under the compile cache (the bf16 copies, the
    quantized checkpoints and the pre-sharded weight store, whose entry names carry the revision).

    The NEFF/HLO caches are shared by every model and keyed by content, so they cannot be told
    apart by recipe; they are left alone, never evicted or pre-read for someone else's model."""
    files = pagecache.files_under([H.snapshot_dir(model_id, revision),
                                   cache / "weights-bf16" / revision])
    short = revision[:8]
    for root in (cache / "quantized", cache / "neuron" / "weights"):
        if root.is_dir():
            files += pagecache.files_under(p for p in root.iterdir() if short in p.name)
    return sorted(set(files))


@dataclass
class RecipeRun:
    plan: Plan
    config: RunConfig
    suite: str | None = None
    variant: str | None = None
    variant_args: dict[str, Any] | None = None
    #: A repro reuses the fingerprint its record had, so it lands next to the original.
    recorded_fingerprint: str | None = None
    run_id: str = field(default_factory=results.new_run_id)

    def __post_init__(self) -> None:
        self.device = self.config.device_name or device_slug()
        recipe = self.plan.recipe
        self.recipes_file = Path(recipe.source) if recipe.source else RECIPES_FILE
        overridden = {}
        if self.plan.passthrough:
            overridden["passthrough"] = self.plan.passthrough
        if recipe.source and self.recipes_file.resolve() != RECIPES_FILE.resolve():
            # A recipe from another file is not the committed recipe of that name.
            overridden["recipes_file"] = file_sha256(self.recipes_file)
        self.fingerprint = self.recorded_fingerprint or (
            fingerprint(overridden) if overridden else None)
        self.dir_name = results.recipe_dir_name(recipe.name.split("@")[0], suite=self.suite,
                                                variant=self.variant,
                                                fingerprint=self.fingerprint)
        root = self.config.results_root or results.results_root()
        self.input_path, self.result_path = results.result_paths(root, self.device,
                                                                 self.dir_name, self.run_id)
        self.work = H.bench_root() / "runs" / self.run_id
        self.cache = H.compile_cache_root()
        self.caller_env = dict(os.environ)
        self.launches: list[dict[str, Any]] = []
        self.samples: dict[str, list[float]] = {m: [] for m in METRICS}
        self.invalid: dict[str, list[str]] = {m: [] for m in METRICS}
        self.extra: dict[str, dict[str, Any]] = {m: {} for m in METRICS}
        self.ran: set[str] = set()
        self.aborted: str | None = None
        self.quality_dump: Path | None = None
        self.provenance: dict[str, Any] | None = None
        self._weight_files: list[Path] | None = None
        self._counter = 0

    # ----------------------------------------------------------------------- bookkeeping

    def selected(self, metric: str) -> bool:
        return metric in self.config.metrics

    def needs_warm(self) -> bool:
        return bool(WARM_CARRIED & set(self.config.metrics))

    def repeats(self, metric: str) -> int:
        return self.config.repeats or DEFAULT_REPEATS.get(metric, 1)

    def start(self) -> None:
        """Write the input record before anything is launched (it survives a failed compile)."""
        self.provenance = provenance.collect(caller_env=self.caller_env)
        results.write_json(self.input_path, {
            "schema_version": results.SCHEMA_VERSION, "method_version": results.METHOD_VERSION,
            "run_id": self.run_id, "kind": "input", "recipe": self._recipe_record(),
            "requested": {"compile": self.plan.compile_argv, "generate": self.plan.generate_argv},
            "config": self._config_record(), "lane": self._lane_record(),
            "provenance": self.provenance,
        })

    def _recipe_record(self) -> dict[str, Any]:
        r = self.plan.recipe
        return {"name": r.name, "description": r.description, "sha256": r.sha256(),
                "recipes_file": str(self.recipes_file),
                "recipes_file_sha256": file_sha256(self.recipes_file), **r.identity(),
                "suite": self.suite, "variant": self.variant, "variant_args": self.variant_args,
                "passthrough": self.plan.passthrough, "fingerprint": self.fingerprint}

    def _config_record(self) -> dict[str, Any]:
        return {"metrics": list(self.config.metrics), "repeats": self.config.repeats,
                "quality_ref": self.config.quality_ref, "timeout_s": self.config.timeout_s}

    def _lane_record(self) -> dict[str, Any]:
        lane = self.config.lane
        return {"devices": lane.devices, "cores": lane.cores,
                "cpus": H.format_cpu_list(lane.cpus)}

    # ---------------------------------------------------------------------------- launches

    def _check_idle(self, name: str) -> H.IdleReport:
        lane = self.config.lane
        idle = H.check_idle(lane.devices, lane.cpus)
        if not idle.idle:
            raise RunAborted(f"lane not idle before {name}: " + "; ".join(idle.reasons()))
        return idle

    def _launch(self, name: str, command: str, argv: list[str], *, cache_root: Path,
                idle: H.IdleReport | None = None, dump_dir: Path | None = None,
                sample_memory: bool = False) -> dict[str, Any]:
        lane = self.config.lane
        idle = idle or self._check_idle(name)
        timing_dir = self.work / "timing" / name
        extra = {"DIFFLET_BENCH_TIMING_DIR": str(timing_dir),
                 "DIFFLET_COMPILE_CACHE": str(cache_root)}
        if dump_dir is not None:
            extra["DIFFLET_DUMP_LATENTS_DIR"] = str(dump_dir)
        recipe = self.plan.recipe
        child = H.child_env(caller=self.caller_env, env_force=recipe.env_force,
                            env_default=recipe.env_default, cores=lane.cores, cpus=lane.cpus,
                            threads=recipe.threads, extra=extra)
        full = [*argv, "--cache-dir", str(cache_root)]
        before = provenance.host_state(cache_root, lane.devices)
        sampler = None
        if sample_memory:
            # Unpinned: "every CPU but this lane's" would be another lane's CPUs, and a process
            # pinned onto them fails that lane's idle check. Unpinned, it is not a pin conflict.
            sampler = M.Sampler(lane.core_ids)
        if sampler is not None:
            sampler.__enter__()
        try:
            done = L.run(command, full, env=child.env, cpus=lane.cpus,
                         log=self.work / "logs" / f"{name}.log", cwd=self.work,
                         timeout_s=self.config.timeout_s)
        finally:
            if sampler is not None:
                sampler.__exit__(None, None, None)
        after = provenance.host_state(cache_root, lane.devices)
        timeline = T.load(timing_dir)
        engine_args = timeline.args()
        record = {"name": name, "command": command, "launch": done.to_dict(),
                  "idle_check": idle.to_dict(), "env_applied": child.applied,
                  "env_dropped": child.dropped, "env": H.env_snapshot(child.env),
                  "threads": child.threads,
                  "host_before": {k: v for k, v in before.items() if k != "neff_files"},
                  "host_after": {k: v for k, v in after.items() if k != "neff_files"},
                  "new_neffs": provenance.new_neffs(before, after),
                  "evidence": timeline.evidence(), "timing_dir": str(timing_dir),
                  "engine_args": engine_args,
                  "effective_mismatches": (effective_mismatches(self.plan, engine_args)
                                           if command == "generate" else [])}
        if sampler is not None:
            stage_of = {pid: timeline.stage_of(pid) for pid in {s.pid for s in sampler.samples}}
            record["memory"] = M.summarize(sampler.samples, stage_of)
            record["memory"]["sampler_error"] = sampler.error
        self.launches.append(record)
        record["_timeline"] = timeline
        return record

    def _generate_argv(self, name: str) -> tuple[list[str], Path]:
        out = self.work / "outputs" / f"{name}{output_suffix(self.plan)}"
        out.parent.mkdir(parents=True, exist_ok=True)
        return ([*self.plan.generate_argv, "--output", str(out),
                 "--work-dir", str(self.work / "work" / name)], out)

    def weight_files(self) -> list[Path]:
        if self._weight_files is None:
            args = self.plan.recipe.args
            self._weight_files = recipe_files(self.cache, str(args["model-id"]),
                                              str(args["revision"]))
        return self._weight_files

    @staticmethod
    def _failed(record: dict[str, Any]) -> str | None:
        launch = record["launch"]
        if launch["timed_out"]:
            return f"{record['name']} timed out"
        if launch["returncode"] != 0:
            return f"{record['name']} exited {launch['returncode']} (see {launch['log']})"
        return None

    @staticmethod
    def _compiled(record: dict[str, Any]) -> list[str]:
        """Why a launch counts as having compiled. The compile counters every rank reports are
        the evidence; new NEFF files in the shared cache are used only when no rank reported
        counters, since another session may write NEFFs into the same cache meanwhile."""
        reasons, counted = [], False
        for ev in record["evidence"]:
            counters = ev.get("compile_counters")
            if counters is None:
                continue
            counted = True
            if counters.get("TotalCompilations"):
                reasons.append(f"rank {ev.get('rank')} of {ev.get('stage')} compiled "
                               f"{counters['TotalCompilations']} graph(s)")
        if not counted and record["new_neffs"]:
            reasons.append(f"{len(record['new_neffs'])} new NEFF(s) appeared in the cache")
        return reasons

    # ----------------------------------------------------------------------------- metrics

    def compile_metric(self) -> None:
        """``compile`` into a new, empty cache root; the root is removed afterwards."""
        self.ran.add("compile")
        temp = self.work / "compile-cache"
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir(parents=True)
        try:
            record = self._launch("compile", "compile", self.plan.compile_argv, cache_root=temp)
            problems = [p for p in (self._failed(record),) if p]
            if not record["new_neffs"]:
                problems.append("no NEFF was compiled into the empty cache: not a cold compile")
            self.samples["compile"].append(record["launch"]["wall_s"])
            self.invalid["compile"] += problems
            sizes = {part: provenance.dir_bytes(temp / part)
                     for part in ("neuron/neff", "neuron/neff_local", "neuron/hlo",
                                  "neuron/weights", "weights-bf16", "quantized")}
            sizes["total"] = provenance.dir_bytes(temp)
            warmup: dict[str, float] = {}
            for ev in record["evidence"]:
                seconds = T.warmup_seconds(ev.get("phase_seconds") or {})
                if seconds is not None:
                    key = ev.get("stage") or "?"
                    warmup[key] = max(warmup.get(key, 0.0), seconds)  # the slowest rank
            self.extra["compile"] = {"cache_bytes": sizes, "warmup_forward_s_by_stage": warmup,
                                     "new_neffs": len(record["new_neffs"])}
        finally:
            shutil.rmtree(temp, ignore_errors=True)  # a full cache: never leave it behind

    def prepare(self) -> None:
        """Compile into the persistent cache (a no-op on a warm cache); not a metric.

        When that compile did compile something, one untimed generate (``prime``) follows:
        ``difflet compile`` does not produce every graph a generate uses (FLUX at 1024: 24 more,
        compiled by the first generate, 2026-11-15), and no sample may pay for them. What the
        prime still had to compile is noted in its launch record."""
        record = self._launch("prepare", "compile", self.plan.compile_argv, cache_root=self.cache)
        failed = self._failed(record)
        if failed:
            raise RunAborted(f"preparing the compile cache failed: {failed}")
        if self.provenance is not None:
            self.provenance["cache_manifests"] = provenance.cache_manifests(
                self.cache, str(self.plan.recipe.args["model-id"]))
        if not self._compiled(record):
            return
        argv, _ = self._generate_argv("prime")
        prime = self._launch("prime", "generate", argv, cache_root=self.cache)
        failed = self._failed(prime)
        if failed:
            raise RunAborted(f"the priming generate failed: {failed}")
        compiled = self._compiled(prime)
        if compiled:
            prime.setdefault("notes", []).append(
                "difflet compile left graphs for the first generate to compile: "
                + "; ".join(compiled))

    def sample(self, metric: str) -> None:
        """One launch of ``e2e_cold`` or ``e2e_warm`` (which also carries dit_step and memory)."""
        self._counter += 1
        name = f"{metric}_{self._counter}"
        idle = self._check_idle(name)  # before touching anyone's page cache
        self.ran.add(metric)
        if metric == "e2e_warm":
            self.ran.update(WARM_CARRIED & set(self.config.metrics))
        files = self.weight_files()
        argv, out = self._generate_argv(name)
        before_resident, total = pagecache.residency(files)
        if metric == "e2e_cold":
            pagecache.evict(files)
        else:
            pagecache.preread(files)
        resident, total = pagecache.residency(files)
        fraction = resident / total if total else 0.0
        record = self._launch(name, "generate", argv, cache_root=self.cache, idle=idle,
                              sample_memory=metric == "e2e_warm" and self.selected("memory"))
        record["page_cache"] = {
            "resident_fraction_before": round(before_resident / total, 4) if total else 0.0,
            "resident_fraction_at_launch": round(fraction, 4), "files": len(files),
            "bytes": total, "excluded": "the shared NEFF/HLO caches (not keyed by recipe)"}
        record["output"] = {"path": str(out), "sha256": _sha256(out)}
        problems = [p for p in (self._failed(record),) if p]
        compiled = self._compiled(record)
        if compiled:
            problems.append(f"{name} compiled: " + "; ".join(compiled))
        elif record["new_neffs"]:
            record.setdefault("notes", []).append(
                f"{len(record['new_neffs'])} NEFF(s) appeared in the shared cache during the "
                "launch, but no rank of this run compiled")
        if record["effective_mismatches"]:
            problems.append(f"{name}: " + "; ".join(record["effective_mismatches"]))
        wall = record["launch"]["wall_s"]
        if metric == "e2e_cold":
            verified = fraction <= RESIDENT_MAX_AFTER_EVICT
            state = "evicted" if verified else "evict_requested_unverified"
            record["cache_state"] = {"page_cache": state, "compile_cache": "warm (persistent)",
                                     "new_process": True}
            self.invalid["e2e_cold"] += problems
            if verified and not problems:
                self.samples["e2e_cold"].append(wall)
            else:
                self.extra["e2e_cold"].setdefault("excluded", []).append(
                    {"launch": name, "wall_s": wall, "resident_fraction": fraction})
            self.extra["e2e_cold"].setdefault("fs_read_bytes", []).append(
                record["launch"]["fs_read_bytes"])
            return
        record["cache_state"] = {"page_cache": "pre-read", "compile_cache": "warm (persistent)",
                                 "new_process": True}
        warm_problems = list(problems)
        for ev in record["evidence"]:
            if ev.get("unwarmed_shapes"):
                warm_problems.append(f"{name}: unwarmed shapes {ev['unwarmed_shapes']} on rank "
                                     f"{ev.get('rank')} of {ev.get('stage')}")
        if fraction < RESIDENT_MIN_AFTER_PREREAD:
            self.extra["e2e_warm"].setdefault("notes", []).append(
                f"{name}: only {fraction:.0%} of the weights were page-cached at launch")
        if self.selected("e2e_warm"):
            self.invalid["e2e_warm"] += warm_problems
            if not warm_problems:
                self.samples["e2e_warm"].append(wall)
            self.extra["e2e_warm"].setdefault("stage_breakdown", []).append(
                T.stage_breakdown(record["_timeline"]))
        if not warm_problems:
            self._dit_step(record)
            self._memory(record)
        else:
            for metric_name in ("dit_step", "memory"):
                if self.selected(metric_name):
                    self.invalid[metric_name].append(f"{name} was not a valid warm run")

    def _dit_step(self, record: dict[str, Any]) -> None:
        if not self.selected("dit_step"):
            return
        timeline: T.Timeline = record["_timeline"]
        per_rank: dict[str, float] = {}
        for rank in timeline.ranks():
            deltas = [d for stamps in timeline.steps(rank=rank) for d in T.deltas(stamps)]
            if deltas:
                per_rank[str(rank)] = statistics.median(deltas)
        stamps_by_request = timeline.steps(rank=0)
        deltas = [d for stamps in stamps_by_request for d in T.deltas(stamps)]
        if not deltas:
            self.invalid["dit_step"].append(f"{record['name']}: no step events on rank 0")
            return
        self.samples["dit_step"] += deltas
        forwards = [f for fs in timeline.forwards(rank=0) for f in fs]
        extra = self.extra["dit_step"]
        extra.setdefault("forward_s", []).extend(forwards)
        extra.setdefault("median_by_rank", []).append(per_rank)
        extra.setdefault("steps_per_request", []).append([len(s) for s in stamps_by_request])
        sync = {ev.get("sync") for ev in record["evidence"] if ev.get("sync")}
        extra["sync"] = sorted(sync)
        for denoise in timeline.spans_named("denoise", rank=0):
            stamps = [t for s in stamps_by_request for t in s
                      if denoise.end_ns and denoise.begin_ns <= t <= denoise.end_ns]
            if not stamps or not denoise.seconds:
                continue
            covered = (stamps[-1] - denoise.begin_ns) / 1e9
            gap = abs(denoise.seconds - covered) / denoise.seconds
            extra.setdefault("self_check_gap", []).append(round(gap, 5))
            if gap > STEP_SUM_TOLERANCE:
                self.invalid["dit_step"].append(
                    f"{record['name']}: the steps cover {covered:.3f} s of a {denoise.seconds:.3f}"
                    f" s denoise loop ({gap:.1%} apart, limit {STEP_SUM_TOLERANCE:.0%})")
        if "self_check_gap" not in extra and "unchecked: no denoise span" not in extra.get(
                "flags", []):
            # FLUX's loop runs inside diffusers' pipe(); without a denoise span the step sum
            # cannot be checked, and the result says so where a reader looks.
            extra.setdefault("flags", []).append("unchecked: no denoise span")
        series = sorted(deltas)
        median = statistics.median(series)
        tail = series[int(round(0.9 * (len(series) - 1)))] / median if median > 0 else None
        flags = extra.setdefault("flags", [])
        if tail and tail > STEP_TAIL_RATIO:
            flags.append(f"p90/median {tail:.2f} > {STEP_TAIL_RATIO}")
        if len(deltas) >= 20:
            first, last = statistics.median(deltas[:10]), statistics.median(deltas[-10:])
            drift = abs(last - first) / first if first > 0 else 0.0
            if drift > STEP_DRIFT:
                flags.append(f"drift {drift:.1%} between the first and last 10 steps")

    def _memory(self, record: dict[str, Any]) -> None:
        if not self.selected("memory"):
            return
        extra = self.extra["memory"]
        sampled = record.get("memory") or {}
        if sampled.get("sampler_error"):
            self.invalid["memory"].append(sampled["sampler_error"])
        peak = sampled.get("peak_gib")
        if peak is not None:
            self.samples["memory"].append(peak)
        extra.setdefault("per_launch", []).append(sampled)
        in_process = [ev.get("device_memory") for ev in record["evidence"]
                      if ev.get("device_memory")]
        weights = [ev["weight_bytes"] for ev in record["evidence"] if ev.get("weight_bytes")]
        if weights:
            extra.setdefault("dit_weight_gib_per_rank", []).append(
                round(max(weights) / (1 << 30), 4))
        if in_process:
            ip_peak = max(m["max_allocated_gib"] for m in in_process)
            extra.setdefault("in_process_max_allocated_gib", []).append(round(ip_peak, 4))
            if peak and abs(peak - ip_peak) / max(peak, ip_peak) > MEMORY_DISAGREEMENT:
                extra.setdefault("flags", []).append(
                    f"{record['name']}: neuron-monitor peak {peak:.2f} GiB and the in-process "
                    f"max_memory_allocated {ip_peak:.2f} GiB disagree by more than "
                    f"{MEMORY_DISAGREEMENT:.0%}")
        # ru_maxrss of the tree is the largest single process's peak, not their sum.
        extra.setdefault("largest_process_rss_gib", []).append(
            round(record["launch"]["max_rss_kib"] / (1 << 20), 3))
        extra.setdefault("fs_read_bytes", []).append(record["launch"]["fs_read_bytes"])

    def quality(self) -> None:
        """One generate with the latent and frame dumps on; compared with the reference."""
        from difflet.bench import quality as Q

        self.ran.add("quality")
        self._counter += 1
        name = f"quality_{self._counter}"
        dump = self.work / "dump" / name
        argv, out = self._generate_argv(name)
        record = self._launch(name, "generate", argv, cache_root=self.cache, dump_dir=dump)
        failed = self._failed(record)
        extra = self.extra["quality"]
        extra["dump"] = Q.dump_summary(dump) if dump.exists() else None
        extra["output"] = {"path": str(out), "sha256": _sha256(out),
                           "note": "the encoded file depends on the encoder build; not compared"}
        if failed:
            self.invalid["quality"].append(failed)
            return
        if not out.exists():
            # A writer that cannot encode falls back to a .pt and exits 0; its frame dump then
            # describes a file that was never written.
            self.invalid["quality"].append(f"the engine wrote no {out.name} (encoder fallback?)")
            return
        if record["effective_mismatches"]:
            self.invalid["quality"].append("; ".join(record["effective_mismatches"]))
        self.quality_dump = dump
        try:
            ref_dir, reference = self._reference(self.config.quality_ref)
            extra["reference"] = reference
            if reference["kind"] == "none":
                extra["not_applicable"] = "dump only (--quality-ref none)"
                return
            if ref_dir is not None:
                extra["comparison"] = Q.compare_dumps(ref_dir, dump)
        except Exception as exc:  # noqa: BLE001 - a bad reference must not lose the record
            self.invalid["quality"].append(f"{type(exc).__name__}: {exc}")

    def _reference(self, ref: str | None) -> tuple[Path | None, dict[str, Any]]:
        if ref is None:
            self.invalid["quality"].append(
                "no quality reference: pass --quality-ref (cpu_fp32, none, or a result JSON), "
                "or give the suite a compare.control or quality_ref")
            return None, {"kind": "missing"}
        if ref == "none":
            return None, {"kind": "none"}
        if ref == "cpu_fp32":
            from difflet.bench import reference

            found = reference.find_reference(self.plan, self.quality_dump)
            if found is None:
                self.invalid["quality"].append(
                    "no frozen cpu_fp32 reference for this recipe and initial latents; build one "
                    "with `difflet bench reference --from <a result with a quality dump>`")
                return None, {"kind": "cpu_fp32"}
            manifest = found / "manifest.json"
            return found, {"kind": "cpu_fp32", "dir": str(found),
                           "reference_id": reference.verify(found)["reference_id"],
                           "manifest_sha256": file_sha256(manifest)}
        path = Path(ref)
        if path.is_dir():
            return path, {"kind": "control", "dir": str(path)}
        record = results.read_json(path)
        dump = (((record.get("metrics") or {}).get("quality") or {}).get("dump") or {}).get("dir")
        if not dump or not Path(dump).is_dir():
            self.invalid["quality"].append(f"{path} has no quality dump on this host")
            return None, {"kind": "control", "result": str(path)}
        return Path(dump), {"kind": "control", "dir": dump, "result": str(path),
                            "run_id": record.get("run_id")}

    # ------------------------------------------------------------------------------ result

    def finish(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for metric in self.config.metrics:
            extra = dict(self.extra.get(metric) or {})
            na = extra.pop("not_applicable", None)
            unit = {"memory": "GiB", "quality": ""}.get(metric, "s")
            samples = self.samples[metric] if metric != "quality" else None
            invalid = list(dict.fromkeys(self.invalid[metric]))
            if metric not in self.ran:
                invalid.append("not run" + (" (the run was aborted)" if self.aborted else ""))
            elif metric != "quality" and not self.samples[metric] and not invalid and not na:
                invalid.append("no valid sample")
            metrics[metric] = results.metric_record(samples=samples, unit=unit, invalid=invalid,
                                                    not_applicable=na, **extra)
        valid = self.aborted is None and all(
            m["valid"] or m["not_applicable"] for m in metrics.values())
        launches = [{k: v for k, v in r.items() if k != "_timeline"} for r in self.launches]
        record = {
            "schema_version": results.SCHEMA_VERSION, "method_version": results.METHOD_VERSION,
            "run_id": self.run_id, "kind": "result", "device": self.device,
            "recipe": self._recipe_record(), "config": self._config_record(),
            "lane": self._lane_record(),
            "requested": {"compile": self.plan.compile_argv, "generate": self.plan.generate_argv},
            "metrics": metrics, "valid": valid, "aborted": self.aborted,
            "launches": launches, "provenance": self.provenance,
            "work_dir": str(self.work), "repro": self.repro_command(),
        }
        results.write_json(self.result_path, record)
        return record

    def repro_command(self) -> str:
        """A command that reruns exactly this record (recipe, overrides, metrics, repeats and
        reference come from the record itself); the lane is host-specific and is restated."""
        lane = self.config.lane
        return shlex.join([sys.executable, "-m", "difflet.cli.main", "bench", "run", "--repro",
                           str(self.result_path), "--devices", ",".join(map(str, lane.devices)),
                           "--cpus", H.format_cpu_list(lane.cpus)])


def _sha256(path: Path) -> str | None:
    from difflet.utils.bench_timing import file_sha256 as hash_file

    return hash_file(str(path)) if Path(path).exists() else None


def run_recipe(run: RecipeRun) -> dict[str, Any]:
    """All of one recipe's selected metrics, in order, then the result record (always)."""
    run.start()
    interrupted: BaseException | None = None
    try:
        if run.selected("compile"):
            run.compile_metric()
        if run.needs_warm() or run.selected("e2e_cold") or run.selected("quality"):
            run.prepare()
        if run.selected("e2e_cold"):
            for _ in range(run.repeats("e2e_cold")):
                run.sample("e2e_cold")
        if run.needs_warm():
            for _ in range(run.repeats("e2e_warm")):
                run.sample("e2e_warm")
        if run.selected("quality"):
            run.quality()
    except (RunAborted, H.HostError) as exc:
        run.aborted = str(exc)
    except Exception as exc:  # noqa: BLE001 - written as an aborted result, then re-raised
        run.aborted = f"{type(exc).__name__}: {exc}"
        interrupted = exc
    except BaseException as exc:  # Ctrl-C, SIGTERM (raised as KeyboardInterrupt by the CLI)
        run.aborted = f"interrupted ({type(exc).__name__})"
        interrupted = exc
    record = run.finish()
    if interrupted is not None:
        raise interrupted
    return record
