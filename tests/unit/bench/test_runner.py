"""The runner's rules, with the launcher, host and page cache faked (no device, no engine)."""
from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from difflet.bench import host as H
from difflet.bench import launch as L
from difflet.bench import pagecache, provenance, results, suite
from difflet.bench import runner as RN
from difflet.bench import timing as T
from difflet.bench.recipe import Recipe

REV = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A fake host: every call is logged in ``calls``; each launch returns ``next_evidence``."""
    state = SimpleNamespace(calls=[], idle=True, evidence=[], new_neffs=[], raise_in_run=None)
    monkeypatch.setattr(H, "bench_root", lambda: tmp_path / "bench")
    monkeypatch.setattr(H, "compile_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(H, "child_env", lambda **kw: H.ChildEnv(env={}, applied={}, dropped=[],
                                                                  threads=None))

    def check_idle(devices, cpus):
        state.calls.append("idle")
        report = H.IdleReport(devices=devices, cpus=cpus, device_holders={d: [] for d in devices})
        if not state.idle:
            report.cpu_busy = {cpus[0]: 0.9}
        return report

    def run(command, argv, **kw):
        state.calls.append(f"launch:{command}")
        if state.raise_in_run:
            raise state.raise_in_run
        if "--output" in argv:
            Path(argv[argv.index("--output") + 1]).write_bytes(b"out")
        return L.Launch(command=command, argv=list(argv), launch_id="x", returncode=0, wall_s=2.0,
                        started_ns=0, ended_ns=1, log="log", fs_read_bytes=0, max_rss_kib=1024,
                        user_s=0.0, sys_s=0.0)

    monkeypatch.setattr(H, "check_idle", check_idle)
    monkeypatch.setattr(L, "run", run)
    monkeypatch.setattr(T, "load", lambda d: T.Timeline(events=list(state.evidence), spans=[],
                                                        processes={}))
    monkeypatch.setattr(provenance, "collect", lambda caller_env: {"git": {}})
    monkeypatch.setattr(provenance, "host_state", lambda cache, devices: {"neff_files": []})
    monkeypatch.setattr(provenance, "new_neffs", lambda before, after: list(state.new_neffs))
    monkeypatch.setattr(provenance, "cache_manifests", lambda cache, model: [])
    monkeypatch.setattr(RN, "recipe_files", lambda cache, model, rev: [])
    monkeypatch.setattr(RN, "effective_mismatches", lambda plan, args: [])
    for name in ("evict", "preread"):
        monkeypatch.setattr(pagecache, name, lambda files, n=name: state.calls.append(n))
    monkeypatch.setattr(pagecache, "residency",
                        lambda files: (state.calls.append("residency"), (0, 0))[1])
    return state


def _run(tmp_path, metrics, quality_ref=None) -> RN.RecipeRun:
    recipe = Recipe(name="flux", args={"model-id": "org/m", "revision": REV, "steps": 4})
    plan = SimpleNamespace(recipe=recipe, passthrough=[], compile_argv=["--model-id", "org/m"],
                           generate_argv=["--model-id", "org/m"])
    config = RN.RunConfig(metrics=list(metrics), lane=RN.Lane(devices=[8], cpus=[0, 1]),
                          quality_ref=quality_ref, results_root=tmp_path / "results",
                          device_name="trn3")
    return RN.RecipeRun(plan=plan, config=config)


def test_recipe_files_are_this_recipes_only(tmp_path, monkeypatch):
    snap = tmp_path / "hub" / "snap"
    cache = tmp_path / "cache"
    for path in (snap / "s.safetensors", cache / "weights-bf16" / REV / "dit" / "a.bin",
                 cache / "weights-bf16" / ("f" * 40) / "dit" / "b.bin",
                 cache / "quantized" / f"org--m__{REV[:8]}" / "dit" / "x.bin",
                 cache / "quantized" / "org--m__ffffffff" / "dit" / "y.bin",
                 cache / "neuron" / "weights" / f"flux-{REV[:8]}-tp4" / "z.bin",
                 cache / "neuron" / "neff" / "shared.neff"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"w")
    monkeypatch.setattr(H, "snapshot_dir", lambda model, rev: snap)
    names = {p.name for p in RN.recipe_files(cache, "org/m", REV)}
    assert names == {"s.safetensors", "a.bin", "x.bin", "z.bin"}  # no other model, no NEFFs


def test_the_idle_check_comes_before_the_page_cache_is_touched(tmp_path, host):
    run = _run(tmp_path, ["e2e_cold"])
    run.start()
    run.sample("e2e_cold")
    assert host.calls[:3] == ["idle", "residency", "evict"]
    assert host.calls.count("idle") == 1 and "launch:generate" in host.calls
    assert run.samples["e2e_cold"] == [2.0]


def test_a_busy_lane_aborts_and_still_writes_the_result(tmp_path, host):
    host.idle = False
    run = _run(tmp_path, ["e2e_cold"])
    record = RN.run_recipe(run)
    assert "not idle" in record["aborted"] and record["valid"] is False
    assert "launch:compile" not in host.calls  # nothing ran on a busy lane
    assert any("not run" in r for r in record["metrics"]["e2e_cold"]["invalid_reasons"])
    assert results.read_json(run.result_path)["aborted"] == record["aborted"]


def test_an_interrupt_writes_the_result_then_propagates(tmp_path, host):
    host.raise_in_run = KeyboardInterrupt()
    run = _run(tmp_path, ["e2e_warm"])
    with pytest.raises(KeyboardInterrupt):
        RN.run_recipe(run)
    assert results.read_json(run.result_path)["aborted"].startswith("interrupted")


def test_a_warm_run_that_compiled_is_invalid(tmp_path, host):
    host.evidence = [{"event": "evidence", "t_ns": 0, "rank": 0, "stage": "dit",
                      "compile_counters": {"TotalCompilations": 2}}]
    run = _run(tmp_path, ["e2e_warm"])
    run.start()
    run.sample("e2e_warm")
    assert not run.samples["e2e_warm"]
    assert any("compiled 2 graph" in r for r in run.invalid["e2e_warm"])


def test_a_neff_from_another_session_is_only_a_note_when_no_rank_compiled(tmp_path, host):
    host.evidence = [{"event": "evidence", "t_ns": 0, "rank": 0, "stage": "dit",
                      "compile_counters": {"TotalCompilations": 0}}]
    host.new_neffs = ["other-model.neff"]
    run = _run(tmp_path, ["e2e_warm"])
    run.start()
    run.sample("e2e_warm")
    assert run.samples["e2e_warm"] == [2.0] and not run.invalid["e2e_warm"]
    assert "no rank of this run compiled" in run.launches[-1]["notes"][0]


@pytest.mark.parametrize("compiled, primed", [(3, True), (0, False)])
def test_prepare_primes_the_cache_only_when_it_compiled(tmp_path, host, compiled, primed):
    host.evidence = [{"event": "evidence", "t_ns": 0, "rank": 0, "stage": "dit",
                      "compile_counters": {"TotalCompilations": compiled}}]
    run = _run(tmp_path, ["e2e_warm"])
    run.start()
    run.prepare()
    names = [launch["name"] for launch in run.launches]
    assert names == (["prepare", "prime"] if primed else ["prepare"])
    if primed:  # the prime compiled too (same fake evidence): noted, not a failure
        assert "first generate to compile" in run.launches[-1]["notes"][0]


def test_quality_without_a_reference_is_invalid(tmp_path, host):
    record = RN.run_recipe(_run(tmp_path, ["quality"], quality_ref=None))
    quality = record["metrics"]["quality"]
    assert not quality["valid"] and any("no quality reference" in r
                                        for r in quality["invalid_reasons"])


def test_the_repro_command_reruns_the_record(tmp_path, host):
    run = _run(tmp_path, ["e2e_warm"])
    command = run.repro_command()
    assert "bench run --repro" in command and str(run.result_path) in command
    assert "--devices 8" in command and "--cpus 0-1" in command


def _suite_file(tmp_path, body):
    path = tmp_path / "s.yaml"
    path.write_text(textwrap.dedent(body))
    return str(path)


@pytest.fixture
def fake_suite(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "plan", lambda recipe: recipe)
    monkeypatch.setattr(suite, "_identity", lambda s: {"suite": s.name})
    monkeypatch.setattr(H, "bench_root", lambda: tmp_path / "bench")
    monkeypatch.setattr(RN, "RecipeRun", lambda **kw: SimpleNamespace(**kw, quality_dump=None,
                                                                       result_path=tmp_path / "r"))


def test_abba_resume_reruns_both_arms_unless_both_are_done(tmp_path, monkeypatch, fake_suite):
    pairs = []
    monkeypatch.setattr(suite, "_abba", lambda s, a, b, *rest: pairs.append(
        (a.plan.name, b.plan.name)) or 0)
    s = suite.load_suite("quant_ab")
    path, manifest = suite.open_experiment(s, "e", resume=False)
    suite.record(path, manifest, "flux/bf16", {"valid": True, "run_id": "r"}, tmp_path / "r")
    suite.run_suite(s, name="e", chosen=s.runs, resume=True, lane=None, timeout_s=1,
                    results_root=None)
    assert pairs == [("qwen_image@quant_ab.bf16", "qwen_image@quant_ab.mxfp8"),
                     ("flux@quant_ab.bf16", "flux@quant_ab.mxfp8")]
    _, manifest = suite.open_experiment(s, "e", resume=True)
    for key in ("flux/mxfp8", "qwen_image/bf16", "qwen_image/mxfp8"):
        suite.record(path, manifest, key, {"valid": True, "run_id": "r"}, tmp_path / "r")
    pairs.clear()
    suite.run_suite(s, name="e", chosen=s.runs, resume=True, lane=None, timeout_s=1,
                    results_root=None)
    assert pairs == []
    with pytest.raises(suite.RecipeError, match="both arms"):
        suite.run_suite(s, name="e", chosen=suite.select(s, "^flux/bf16$"), resume=True,
                        lane=None, timeout_s=1, results_root=None)


def test_a_sequential_resume_compares_with_the_earlier_control_dump(tmp_path, monkeypatch,
                                                                     fake_suite):
    seen = {}

    def run_recipe(run):
        seen[run.variant] = run.config.quality_ref
        return {"valid": True, "run_id": "x"}

    monkeypatch.setattr(RN, "run_recipe", run_recipe)
    s = suite.load_suite(_suite_file(tmp_path, """\
        metrics: [quality]
        recipes: [flux]
        variants: {bf16: {}, mxfp8: {args: {quant: mxfp8}}}
        compare: {control: bf16, candidate: mxfp8}
        """))
    dump = tmp_path / "dump"
    dump.mkdir()
    control = tmp_path / "control.json"
    results.write_json(control, {"metrics": {"quality": {"dump": {"dir": str(dump)}}}})
    path, manifest = suite.open_experiment(s, "seq", resume=False)
    suite.record(path, manifest, "flux/bf16", {"valid": True, "run_id": "c"}, control)
    suite.run_suite(s, name="seq", chosen=s.runs, resume=True, lane=None, timeout_s=1,
                    results_root=None)
    assert seen == {"mxfp8": str(dump)}
