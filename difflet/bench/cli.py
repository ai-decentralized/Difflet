"""``difflet bench``: list, run, reference, compare, report (design §4).

Everything after the first ``--`` is passed through to ``difflet compile`` / ``difflet generate``
(routed by the real parsers), so the pass-through is split off before bench parses its own flags.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import sys
import threading
from pathlib import Path
from typing import Sequence

from difflet.bench import host as H
from difflet.bench.recipe import RECIPES_FILE, Recipe, RecipeError, load_recipes, plan


def build_parser() -> argparse.ArgumentParser:
    from difflet.bench.runner import DEFAULT_METRICS, LAUNCH_TIMEOUT_S, METRICS

    p = argparse.ArgumentParser(
        prog="difflet bench",
        description="Benchmark named recipes of difflet compile/generate flags on the neuron "
                    "backend and record every result with its provenance. Flags after `--` go to "
                    "difflet compile/generate. (`difflet run` is something else: download + "
                    "compile + generate once.)")
    sub = p.add_subparsers(dest="action", required=True)

    sub.add_parser("list", help="recipes, suites and metrics available here")

    run = sub.add_parser("run", help="run recipes or a suite and write result JSON")
    what = run.add_mutually_exclusive_group()
    what.add_argument("--recipe", help="recipe name(s), comma-separated, 'all', or a YAML file")
    what.add_argument("--suite", help="suite name or YAML file")
    what.add_argument("--repro", type=Path, help="rerun exactly what a result JSON recorded")
    what.add_argument("--check-env", type=Path, metavar="RESULT_JSON",
                      help="only compare this host's environment with a result's and exit")
    run.add_argument("--metrics", help=f"comma-separated subset of {','.join(METRICS)} "
                                       f"(default {','.join(DEFAULT_METRICS)})")
    run.add_argument("--devices", help="Neuron device indices, e.g. 8 or 8,9")
    run.add_argument("--cpus", help="host CPUs to pin to, e.g. 0-47 (required with --devices)")
    run.add_argument("--repeats", type=int, help="samples per e2e metric (overrides defaults)")
    run.add_argument("--quality-ref", help="cpu_fp32, none (dump only), or a result JSON whose "
                                           "quality dump is the control")
    run.add_argument("--name", help="experiment name for a suite (resume unit)")
    run.add_argument("--select", help="regex over a suite's run ids <recipe>/<variant>")
    run.add_argument("--resume", action="store_true", help="continue a suite experiment")
    run.add_argument("--dry-run", action="store_true",
                     help="print the routed commands, environment and schedule; launch nothing")
    run.add_argument("--timeout", type=float, default=LAUNCH_TIMEOUT_S,
                     help="seconds per launch before it is stopped")
    run.add_argument("--results-dir", type=Path, help="override the results root")
    run.add_argument("--device-name", help="override the results folder (default: from the "
                                           "instance type, e.g. trn3)")

    ref = sub.add_parser("reference", help="build and freeze a CPU fp32 quality reference")
    ref.add_argument("--from", dest="source", type=Path, required=True,
                     help="a result JSON with a quality dump (run with --quality-ref none)")
    ref.add_argument("--cpus", required=True, help="host CPUs to pin the CPU reference to")

    cmp_ = sub.add_parser("compare", help="compare two result JSONs (B against A)")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)

    rep = sub.add_parser("report", help="regenerate SUMMARY.md / REPORT.md from result JSON")
    rep.add_argument("root", type=Path, nargs="?", help="results root (default: the checkout's)")
    rep.add_argument("--all", action="store_true", help="every result, not only the latest")
    return p


def split_passthrough(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    argv = list(argv)
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _interrupt(signum: int, _frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv: Sequence[str] | None = None) -> int:
    head, passthrough = split_passthrough(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(head)
    # SIGTERM/SIGHUP (a closed ssh session, a job scheduler) unwind like Ctrl-C: the launch's
    # processes are stopped and the result is written, marked aborted. SIGKILL of bench cannot
    # be caught: its ranks then keep their NeuronCores until killed by their launch id.
    previous = {}
    if threading.current_thread() is threading.main_thread():
        previous = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        return _main(args, passthrough)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _main(args, passthrough: list[str]) -> int:
    try:
        if args.action == "list":
            return _list()
        if args.action == "run":
            return _run(args, passthrough)
        if passthrough:
            raise RecipeError(f"`--` pass-through only applies to `run`, got {passthrough}")
        if args.action == "reference":
            return _reference(args)
        if args.action == "compare":
            return _compare(args)
        if args.action == "report":
            return _report(args)
    except (RecipeError, H.HostError) as exc:
        print(f"difflet bench: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt as exc:
        print(f"difflet bench: interrupted ({exc or 'Ctrl-C'}); the partial result was written",
              file=sys.stderr)
        return 130
    return 2


def _list() -> int:
    from difflet.bench.runner import DEFAULT_METRICS, METRICS
    from difflet.bench.suite import SUITES_DIR

    recipes = load_recipes()
    print(f"recipes ({RECIPES_FILE}):")
    for name, r in recipes.items():
        shape = "x".join(str(r.args[k]) for k in ("height", "width", "num-frames") if k in r.args)
        print(f"  {name:20s} {r.args.get('model-id')}  {shape or '(model default shape)'}  "
              f"steps={r.args.get('steps')}  tp={r.args.get('tp-degree', 'default')}")
    print(f"suites ({SUITES_DIR}):")
    for path in sorted(SUITES_DIR.glob("*.yaml")):
        print(f"  {path.stem}")
    print(f"metrics: {', '.join(METRICS)} (default: {', '.join(DEFAULT_METRICS)})")
    return 0


def _metrics(text: str | None) -> list[str]:
    from difflet.bench.runner import DEFAULT_METRICS, METRICS

    if not text:
        return list(DEFAULT_METRICS)
    chosen = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in chosen if m not in METRICS]
    if bad:
        raise RecipeError(f"unknown metrics {bad}; choose from {list(METRICS)}")
    return chosen


def _lane(args, *, required: bool):
    from difflet.bench.runner import Lane

    if not args.devices and not args.cpus:
        if required:
            raise RecipeError("--devices and --cpus are required (pick an idle device and the "
                              "CPUs of its NUMA node; bench checks both are idle)")
        return None
    if not (args.devices and args.cpus):
        raise RecipeError("give --devices and --cpus together")
    devices = [int(d) for d in str(args.devices).split(",") if d.strip()]
    return Lane(devices=devices, cpus=H.parse_cpu_list(args.cpus))


def _recipes(spec: str) -> list[Recipe]:
    path = Path(spec)
    if path.suffix in (".yaml", ".yml") and path.exists():
        return list(load_recipes(path).values())
    recipes = load_recipes()
    if spec == "all":
        return list(recipes.values())
    out = []
    for name in (s.strip() for s in spec.split(",")):
        if name not in recipes:
            raise RecipeError(f"unknown recipe {name!r}; known: {sorted(recipes)}")
        out.append(recipes[name])
    return out


def _repro_recipe(path: Path) -> tuple[Recipe, list[str], dict]:
    from difflet.bench import results

    record = results.read_json(path)
    r = record.get("recipe") or {}
    if not r.get("args"):
        raise RecipeError(f"{path} records no recipe args")
    recipe = Recipe(name=r["name"], args=dict(r["args"]), env_force=dict(r.get("env_force") or {}),
                    env_default=dict(r.get("env_default") or {}), threads=r.get("threads"),
                    description=r.get("description") or "")
    if recipe.sha256() != r.get("sha256"):
        raise RecipeError(f"{path}: the recorded recipe does not match its recorded sha256")
    return recipe, list(r.get("passthrough") or []), record


def _check_quality_ref(ref: str | None) -> None:
    """Fail before anything launches, not after an hour of timing runs."""
    if ref in (None, "none", "cpu_fp32"):
        return
    path = Path(ref)
    if not path.exists():
        raise RecipeError(f"--quality-ref {ref}: no such file or directory")
    if path.is_file():
        from difflet.bench import results

        try:
            record = results.read_json(path)
        except ValueError as exc:
            raise RecipeError(f"--quality-ref {ref}: not a result JSON ({exc})") from None
        dump = (((record.get("metrics") or {}).get("quality") or {}).get("dump") or {}).get("dir")
        if not dump or not Path(dump).is_dir():
            raise RecipeError(f"--quality-ref {ref}: the result has no quality dump on this host")


def _run(args, passthrough: list[str]) -> int:
    from difflet.bench import provenance, results
    from difflet.bench import runner as RN

    if args.check_env:
        diffs = provenance.check_env(results.read_json(args.check_env),
                                     provenance.collect(caller_env=dict(os.environ)))
        for d in diffs:
            print(f"differs: {d}")
        print("environment matches" if not diffs else f"{len(diffs)} difference(s)")
        return 0 if not diffs else 1

    if args.suite:
        return _run_suite(args, passthrough)

    repro: dict = {}
    repeats, quality_ref = args.repeats, args.quality_ref
    if args.repro:
        recipe, recorded_pass, record = _repro_recipe(args.repro)
        if passthrough:
            raise RecipeError("--repro reruns the recorded flags; do not add a pass-through")
        passthrough = recorded_pass
        recipes = [recipe]
        config = record.get("config") or {}
        metrics = (_metrics(args.metrics) if args.metrics
                   else list(config.get("metrics") or record.get("metrics") or []))
        repeats = repeats or config.get("repeats")
        quality_ref = quality_ref or config.get("quality_ref")
        r = record.get("recipe") or {}
        repro = {"suite": r.get("suite"), "variant": r.get("variant"),
                 "variant_args": r.get("variant_args"),
                 "recorded_fingerprint": r.get("fingerprint")}
        print(f"[bench] repro of {record.get('run_id')} (recorded at commit "
              f"{((record.get('provenance') or {}).get('git') or {}).get('commit')})", flush=True)
    elif args.recipe:
        recipes = _recipes(args.recipe)
        metrics = _metrics(args.metrics)
    else:
        raise RecipeError("give --recipe, --suite, --repro or --check-env")

    plans = [plan(r, passthrough) for r in recipes]  # validate all before launching any
    if "quality" in metrics and not quality_ref:
        raise RecipeError("--metrics quality needs --quality-ref (cpu_fp32, none for dump only, "
                          "or a result JSON whose quality dump is the control)")
    if "quality" in metrics:
        _check_quality_ref(quality_ref)
    lane = _lane(args, required=not args.dry_run)
    if args.dry_run:
        for p in plans:
            _print_plan(p, metrics, lane, args)
        return 0
    worst = 0
    for p in plans:
        config = RN.RunConfig(metrics=metrics, lane=lane, repeats=repeats,
                              timeout_s=args.timeout, quality_ref=quality_ref,
                              results_root=args.results_dir, device_name=args.device_name)
        run = RN.RecipeRun(plan=p, config=config, **repro)
        result = RN.run_recipe(run)
        _print_result(result, run.result_path)
        worst = max(worst, 2 if result.get("aborted") else (0 if result["valid"] else 1))
    return worst


def _run_suite(args, passthrough: list[str]) -> int:
    from difflet.bench import suite as S

    if passthrough:
        raise RecipeError("a suite's runs are its variants; `--` pass-through does not apply")
    suite = S.load_suite(args.suite)
    chosen = S.select(suite, args.select)
    lane = _lane(args, required=not args.dry_run)
    if args.dry_run:
        for r in chosen:
            print(f"== {r.run_key}")
            _print_plan(plan(r.recipe), suite.metrics, lane, args)
        print(f"order: {suite.order}, repeats: {suite.repeats}, control: {suite.control}, "
              f"candidate: {suite.candidate}")
        return 0
    if not args.name:
        raise RecipeError("a suite run needs --name (the experiment name used by --resume)")
    if args.metrics:
        raise RecipeError("a suite's metrics are its own; edit the suite instead of --metrics")
    if args.repeats:
        suite.repeats = args.repeats
    if args.quality_ref:
        _check_quality_ref(args.quality_ref)
        suite.quality_ref = args.quality_ref
    S.check_quality(suite)
    return S.run_suite(suite, name=args.name, chosen=chosen, resume=args.resume, lane=lane,
                       timeout_s=args.timeout, results_root=args.results_dir,
                       device_name=args.device_name)


def _print_plan(p, metrics: list[str], lane, args) -> None:
    from difflet.bench.launch import command_line

    cpus = lane.cpus if lane else [0]
    print(f"recipe {p.recipe.name}" + (f" (+ pass-through {p.passthrough})" if p.passthrough
                                       else ""))
    print("  compile:  " + shlex.join(command_line("compile", [*p.compile_argv, "--cache-dir",
                                                             "<cache>"], cpus)))
    print("  generate: " + shlex.join(command_line("generate", [*p.generate_argv, "--output",
                                                              "<out>", "--work-dir", "<work>",
                                                              "--cache-dir", "<cache>"], cpus)))
    print(f"  metrics:  {', '.join(metrics)}; repeats: {args.repeats or 'defaults'}")
    if lane is None:
        print("  lane: none given (no idle check; --devices and --cpus are needed to run)")
        return
    child = H.child_env(caller=dict(os.environ), env_force=p.recipe.env_force,
                        env_default=p.recipe.env_default, cores=lane.cores, cpus=lane.cpus,
                        threads=p.recipe.threads)
    print(f"  lane: devices {lane.devices} -> cores {lane.cores}, cpus "
          f"{H.format_cpu_list(lane.cpus)}, threads {child.threads}")
    for name, info in sorted(child.applied.items()):
        print(f"  env {name}={info['value']}  ({info['source']})")
    if child.dropped:
        print(f"  env dropped from the caller: {', '.join(child.dropped)}")
    idle = H.check_idle(lane.devices, lane.cpus)
    print("  idle check: " + ("idle" if idle.idle else "NOT IDLE: " + "; ".join(idle.reasons())))
    if idle.numa_warning:
        print(f"  warning: {idle.numa_warning}")


def _print_result(result: dict, path: Path) -> None:
    status = "valid" if result.get("valid") else ("ABORTED" if result.get("aborted") else "INVALID")
    print(f"[bench] {result['recipe']['name']}: {status} -> {path}", flush=True)
    if result.get("aborted"):
        print(f"[bench]   {result['aborted']}", flush=True)
    for name, metric in (result.get("metrics") or {}).items():
        summary = metric.get("summary") or {}
        value = (f"median {summary['median']:.6g} {metric.get('unit', '')} (n={summary['n']})"
                 if summary.get("n") else (metric.get("not_applicable") or ""))
        reasons = "; ".join(metric.get("invalid_reasons") or [])
        print(f"[bench]   {name}: {value}" + (f"  INVALID: {reasons}" if reasons else ""),
              flush=True)


def _reference(args) -> int:
    from difflet.bench import reference, results

    path = reference.build(results.read_json(args.source), cpus=H.parse_cpu_list(args.cpus))
    print(f"[bench] reference frozen at {path}")
    return 0


def _compare(args) -> int:
    from difflet.bench import compare, results

    try:
        print(compare.render(results.read_json(args.a), results.read_json(args.b)))
    except compare.CompareError as exc:
        print(f"difflet bench compare: {exc}", file=sys.stderr)
        return 1
    return 0


def _report(args) -> int:
    from difflet.bench import report, results

    root = args.root or results.results_root()
    written = report.generate(root, show_all=args.all)
    for path in written:
        print(path)
    if not written:
        print(f"no results under {root}")
    return 0

