"""Suites: a named list of recipe x variant runs, with resume and ABBA ordering (design §6).

A suite file (``difflet/bench/suites/<name>.yaml``, or any path)::

    description: bf16 against MXFP8
    metrics: [compile, e2e_cold, e2e_warm, dit_step, memory, quality]
    recipes: [qwen_image, flux]
    variants:
      bf16: {}
      mxfp8: {args: {quant: mxfp8}}
    compare: {control: bf16, candidate: mxfp8}
    order: ABBA
    repeats: 2

The only product is recipes x variants; each run's id is ``<recipe>/<variant>``. A run of a suite is
an experiment with a name (``--name``); its manifest under ``$DIFFLET_BENCH_ROOT/experiments/``
lists every planned run and where its result went, so ``--resume`` skips the runs already done with
a valid result, provided the commit and the recipe file are the ones the experiment started with.

With ``order: ABBA`` each arm compiles once; then for every timed metric the arms' launches
alternate control, candidate, candidate, control, ``repeats`` rounds, so drift on the host shows up
as a difference between one arm's two launches rather than as a speed-up. The candidate's quality
is measured against the control's dumps from the same experiment.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from difflet.bench import host as H
from difflet.bench import provenance
from difflet.bench.recipe import (RECIPES_FILE, Recipe, RecipeError, file_sha256, load_recipes,
                                  merge_args, plan)

SUITES_DIR = Path(__file__).with_name("suites")
TOP_LEVEL = frozenset({"description", "metrics", "recipes", "variants", "compare", "order",
                       "repeats", "quality_ref"})
ORDERS = ("sequential", "ABBA")


@dataclass
class SuiteRun:
    run_key: str  # "<recipe>/<variant>"
    recipe_name: str
    variant: str
    recipe: Recipe
    variant_args: dict[str, Any]


@dataclass
class Suite:
    name: str
    path: Path
    description: str
    metrics: list[str]
    runs: list[SuiteRun]
    control: str | None
    candidate: str | None
    order: str
    repeats: int | None
    quality_ref: str | None


def suite_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.exists():
        return path
    candidate = SUITES_DIR / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    known = sorted(p.stem for p in SUITES_DIR.glob("*.yaml"))
    raise RecipeError(f"unknown suite {name_or_path!r}; known: {known}")


def load_suite(name_or_path: str, recipes: dict[str, Recipe] | None = None) -> Suite:
    """Parse and validate a suite: every recipe exists, every variant's args pass the checks."""
    import yaml

    from difflet.bench.runner import METRICS

    path = suite_path(name_or_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RecipeError(f"{path}: a suite must be a mapping")
    unknown = set(raw) - TOP_LEVEL
    if unknown:
        raise RecipeError(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(TOP_LEVEL)}")
    recipes = recipes if recipes is not None else load_recipes(RECIPES_FILE)
    metrics = list(raw.get("metrics") or [])
    bad = [m for m in metrics if m not in METRICS]
    if bad or not metrics:
        raise RecipeError(f"{path}: metrics must be a non-empty subset of {list(METRICS)}; got {bad}")
    variants = raw.get("variants") or {"default": {}}
    if not isinstance(variants, dict):
        raise RecipeError(f"{path}: variants must be a mapping of name -> {{args: ...}}")
    runs: list[SuiteRun] = []
    for recipe_name in raw.get("recipes") or []:
        if recipe_name not in recipes:
            raise RecipeError(f"{path}: unknown recipe {recipe_name!r}")
        for variant, body in variants.items():
            body = body or {}
            if set(body) - {"args"}:
                raise RecipeError(f"{path}: variant {variant!r} may only set args")
            merged = merge_args(recipes[recipe_name], body.get("args") or {},
                                name=f"{recipe_name}@{path.stem}.{variant}")
            plan(merged)  # validate now: a bad variant fails the whole suite before it starts
            runs.append(SuiteRun(run_key=f"{recipe_name}/{variant}", recipe_name=recipe_name,
                                 variant=str(variant), recipe=merged,
                                 variant_args=dict(body.get("args") or {})))
    if not runs:
        raise RecipeError(f"{path}: no runs (recipes x variants is empty)")
    compare = raw.get("compare") or {}
    control, candidate = compare.get("control"), compare.get("candidate")
    for arm in (control, candidate):
        if arm is not None and arm not in variants:
            raise RecipeError(f"{path}: compare names unknown variant {arm!r}")
    order = raw.get("order", "sequential")
    if order not in ORDERS:
        raise RecipeError(f"{path}: order must be one of {ORDERS}")
    if order == "ABBA" and not (control and candidate):
        raise RecipeError(f"{path}: order ABBA needs compare.control and compare.candidate")
    repeats = raw.get("repeats")
    if repeats is not None and (not isinstance(repeats, int) or repeats < 1):
        raise RecipeError(f"{path}: repeats must be a positive integer")
    return Suite(name=path.stem, path=path, description=str(raw.get("description") or ""),
                 metrics=metrics, runs=runs, control=control, candidate=candidate, order=order,
                 repeats=repeats, quality_ref=raw.get("quality_ref"))


def select(suite: Suite, pattern: str | None) -> list[SuiteRun]:
    if not pattern:
        return list(suite.runs)
    regex = re.compile(pattern)
    chosen = [r for r in suite.runs if regex.search(r.run_key)]
    if not chosen:
        raise RecipeError(f"--select {pattern!r} matches none of "
                          f"{[r.run_key for r in suite.runs]}")
    return chosen


# ------------------------------------------------------------------------------ experiments


def experiment_dir(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise RecipeError(f"bad experiment name {name!r} (letters, digits, '.', '_', '-')")
    return H.bench_root() / "experiments" / name


def _identity(suite: Suite) -> dict[str, Any]:
    git = provenance.git_state(H.source_root()) or {}
    return {"suite": suite.name, "suite_sha256": file_sha256(suite.path),
            "recipes_sha256": file_sha256(RECIPES_FILE), "commit": git.get("commit")}


def open_experiment(suite: Suite, name: str, *, resume: bool) -> tuple[Path, dict[str, Any]]:
    directory = experiment_dir(name)
    path = directory / "manifest.json"
    identity = _identity(suite)
    if path.exists():
        if not resume:
            raise RecipeError(f"experiment {name!r} exists ({path}); pass --resume to continue it "
                              "or choose another --name")
        manifest = json.loads(path.read_text())
        for key, value in identity.items():
            if manifest.get(key) != value:
                raise RecipeError(f"cannot resume {name!r}: {key} was {manifest.get(key)!r}, now "
                                  f"{value!r}")
        return path, manifest
    if resume:
        raise RecipeError(f"no experiment {name!r} to resume")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {**identity, "name": name, "runs": {r.run_key: {"status": "pending"}
                                                   for r in suite.runs}}
    _save(path, manifest, create=True)
    return path, manifest


def _save(path: Path, manifest: dict[str, Any], *, create: bool = False) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    if not create:
        tmp.replace(path)
        return
    try:
        os.link(tmp, path)  # atomic and exclusive: two sessions cannot both start one experiment
    except FileExistsError:
        raise RecipeError(f"experiment {path.parent.name!r} was just started by another "
                          "session; choose another --name") from None
    finally:
        tmp.unlink(missing_ok=True)


def done(manifest: dict[str, Any], run_key: str) -> bool:
    entry = manifest["runs"].get(run_key) or {}
    return entry.get("status") == "done" and bool(entry.get("valid"))


def record(path: Path, manifest: dict[str, Any], run_key: str, result: dict[str, Any],
           result_path: Path) -> None:
    manifest["runs"][run_key] = {"status": "done", "valid": bool(result.get("valid")),
                                 "aborted": result.get("aborted"), "result": str(result_path),
                                 "run_id": result.get("run_id")}
    _save(path, manifest)


# --------------------------------------------------------------------------------- running


def check_quality(suite: Suite) -> None:
    """A suite measuring quality must say against what: its control arm or a quality_ref."""
    if "quality" in suite.metrics and not (suite.control or suite.quality_ref):
        raise RecipeError(f"{suite.path}: metrics include quality but the suite has neither "
                          "compare.control nor quality_ref (cpu_fp32, or none for dump only)")


def _control_dump(manifest: dict[str, Any], run_key: str) -> str | None:
    """The quality dump of a control run done in an earlier session of this experiment."""
    from difflet.bench import results

    entry = manifest["runs"].get(run_key) or {}
    if not entry.get("result") or not Path(entry["result"]).exists():
        return None
    metric = (results.read_json(Path(entry["result"])).get("metrics") or {}).get("quality") or {}
    dump = (metric.get("dump") or {}).get("dir")
    return dump if dump and Path(dump).is_dir() else None


def run_suite(suite: Suite, *, name: str, chosen: list[SuiteRun], resume: bool, lane,
              timeout_s: float, results_root: Path | None,
              device_name: str | None = None) -> int:
    """Run (or resume) ``chosen`` runs of ``suite``; returns the bench exit code."""
    check_quality(suite)
    directory = experiment_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released when the file closes
        except BlockingIOError:
            raise RecipeError(f"experiment {name!r} is being run by another session") from None
        return _run_locked(suite, name=name, chosen=chosen, resume=resume, lane=lane,
                           timeout_s=timeout_s, results_root=results_root,
                           device_name=device_name)


def _run_locked(suite: Suite, *, name: str, chosen: list[SuiteRun], resume: bool, lane,
                timeout_s: float, results_root: Path | None, device_name: str | None) -> int:
    from difflet.bench import runner as RN

    manifest_path, manifest = open_experiment(suite, name, resume=resume)

    def make(r: SuiteRun, quality_ref: str | None) -> RN.RecipeRun:
        config = RN.RunConfig(metrics=suite.metrics, lane=lane, repeats=suite.repeats,
                              timeout_s=timeout_s, quality_ref=quality_ref,
                              results_root=results_root, device_name=device_name)
        return RN.RecipeRun(plan=plan(r.recipe), config=config, suite=suite.name,
                            variant=r.variant, variant_args=r.variant_args)

    def skipped(r: SuiteRun) -> None:
        print(f"[bench] {r.run_key}: done in an earlier session, skipped", flush=True)

    worst = 0
    if suite.order == "ABBA":
        by_recipe: dict[str, dict[str, SuiteRun]] = {}
        for r in chosen:
            by_recipe.setdefault(r.recipe_name, {})[r.variant] = r
        for recipe_name, arms in by_recipe.items():
            if suite.control not in arms or suite.candidate not in arms:
                raise RecipeError(f"ABBA runs both arms of {recipe_name!r} together; --select "
                                  f"must choose {suite.control!r} and {suite.candidate!r}")
            pair = (arms[suite.control], arms[suite.candidate])
            if all(done(manifest, r.run_key) for r in pair):
                for r in pair:
                    skipped(r)
                continue
            # The arms are measured interleaved: one done without the other is rerun with it.
            a = make(pair[0], suite.quality_ref or "none")
            b = make(pair[1], None)
            worst = max(worst, _abba(suite, a, b, manifest_path, manifest, arms))
        return worst
    control_dumps: dict[str, str | None] = {}
    if suite.control:
        for r in chosen:
            if r.variant == suite.control and done(manifest, r.run_key):
                control_dumps[r.recipe_name] = _control_dump(manifest, r.run_key)
    for r in sorted(chosen, key=lambda r: r.variant != suite.control):  # controls first
        if done(manifest, r.run_key):
            skipped(r)
            continue
        ref = suite.quality_ref
        if suite.control and r.variant == suite.control:
            ref = suite.quality_ref or "none"
        elif suite.control:
            # No dump from the control (not run, or it failed): None makes quality invalid.
            ref = control_dumps.get(r.recipe_name)
        run = make(r, ref)
        result = RN.run_recipe(run)  # re-raises an interrupt once the result is written
        if r.variant == suite.control:
            control_dumps[r.recipe_name] = str(run.quality_dump) if run.quality_dump else None
        record(manifest_path, manifest, r.run_key, result, run.result_path)
        worst = max(worst, _exit_code(result))
        print(f"[bench] {r.run_key}: {'valid' if result['valid'] else 'INVALID'} -> "
              f"{run.result_path}", flush=True)
    return worst


def _exit_code(result: dict[str, Any]) -> int:
    return 2 if result.get("aborted") else (0 if result.get("valid") else 1)


def _control_drift(a) -> int:
    """How many distinct outputs the control's own timed launches produced (1: reproducible)."""
    hashes = {(rec.get("output") or {}).get("sha256") for rec in a.launches
              if rec.get("command") == "generate" and (rec.get("output") or {}).get("sha256")}
    return len(hashes)


def _abba(suite: Suite, a, b, manifest_path: Path, manifest: dict[str, Any],
          arms: dict[str, SuiteRun]) -> int:
    from difflet.bench import runner as RN

    pair = (a, b)
    interrupted: BaseException | None = None
    try:
        for run in pair:
            run.start()
        for run in pair:
            if run.selected("compile"):
                run.compile_metric()
        for run in pair:
            run.prepare()
        rounds = suite.repeats or 1
        for metric in ("e2e_cold", "e2e_warm"):
            wanted = a.selected(metric) or (metric == "e2e_warm" and a.needs_warm())
            if not wanted:
                continue
            for _ in range(rounds):
                for run in (a, b, b, a):
                    run.sample(metric)
        if a.selected("quality"):
            a.quality()
            if a.quality_dump is not None:
                b.config.quality_ref = str(a.quality_dump)
                b.quality()
            else:
                b.invalid["quality"].append("the control arm produced no quality dump")
    except (RN.RunAborted, H.HostError) as exc:
        for run in pair:
            run.aborted = str(exc)
    except BaseException as exc:  # an error or an interrupt: both records are still written
        for run in pair:
            run.aborted = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else (
                f"interrupted ({type(exc).__name__})")
        interrupted = exc
    distinct = _control_drift(a)
    if distinct > 1:
        reason = (f"the control's own outputs differ between its launches ({distinct} distinct "
                  "hashes): its baseline is not reproducible, so the A/B difference could be noise")
        for metric in b.config.metrics:
            b.invalid[metric].append(reason)
    worst = 0
    for run, arm in ((a, suite.control), (b, suite.candidate)):
        result = run.finish()
        record(manifest_path, manifest, arms[arm].run_key, result, run.result_path)
        worst = max(worst, _exit_code(result))
        print(f"[bench] {arms[arm].run_key}: {'valid' if result['valid'] else 'INVALID'} -> "
              f"{run.result_path}", flush=True)
    if interrupted is not None:
        raise interrupted
    return worst
