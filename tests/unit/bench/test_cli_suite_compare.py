"""The bench command line, suites, compare and report."""
from __future__ import annotations

import importlib
import textwrap

import pytest

from difflet.bench import cli, compare, report, results, suite


def test_difflet_bench_is_dispatched_before_the_engine_parser(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    main = importlib.import_module("difflet.cli.main")
    with pytest.raises(SystemExit) as exc:
        main.main(["bench", "run", "--recipe", "flux", "--", "--steps", "8"])
    assert exc.value.code == 0 or exc.value.code is None or seen
    assert seen["argv"] == ["run", "--recipe", "flux", "--", "--steps", "8"]


def test_split_passthrough():
    assert cli.split_passthrough(["run", "--recipe", "a", "--", "--steps", "8", "--", "x"]) == (
        ["run", "--recipe", "a"], ["--steps", "8", "--", "x"])
    assert cli.split_passthrough(["list"]) == (["list"], [])


def test_quality_needs_a_reference_flag(capsys):
    code = cli.main(["run", "--recipe", "flux", "--metrics", "quality", "--dry-run"])
    assert code == 2 and "--quality-ref" in capsys.readouterr().err


def test_unknown_metric_and_recipe_are_refused(capsys):
    assert cli.main(["run", "--recipe", "flux", "--metrics", "speed", "--dry-run"]) == 2
    assert cli.main(["run", "--recipe", "nope", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "unknown metrics" in err and "unknown recipe 'nope'" in err


def test_devices_and_cpus_go_together(capsys):
    assert cli.main(["run", "--recipe", "flux", "--devices", "8"]) == 2
    assert "together" in capsys.readouterr().err


def _suite(tmp_path, body):
    path = tmp_path / "s.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_shipped_suite_loads_and_expands(monkeypatch):
    monkeypatch.setattr(suite, "plan", lambda recipe: None)  # the CLI checks need neuron
    s = suite.load_suite("quant_ab")
    assert [r.run_key for r in s.runs] == ["qwen_image/bf16", "qwen_image/mxfp8",
                                           "flux/bf16", "flux/mxfp8"]
    assert s.order == "ABBA" and s.control == "bf16" and s.repeats == 2
    mx = next(r for r in s.runs if r.run_key == "flux/mxfp8")
    assert mx.recipe.args["quant"] == "mxfp8" and mx.recipe.name == "flux@quant_ab.mxfp8"
    assert [r.run_key for r in suite.select(s, "^flux/")] == ["flux/bf16", "flux/mxfp8"]
    with pytest.raises(suite.RecipeError, match="matches none"):
        suite.select(s, "nothing")


@pytest.mark.parametrize("body, match", [
    ("metrics: [compile]\nrecipes: [flux]\nextra: 1\n", "unknown keys"),
    ("metrics: [speed]\nrecipes: [flux]\n", "subset"),
    ("metrics: [compile]\nrecipes: [nope]\n", "unknown recipe"),
    ("metrics: [compile]\nrecipes: [flux]\norder: ABBA\n", "needs compare"),
    ("metrics: [compile]\nrecipes: [flux]\nvariants: {a: {env: {}}}\n", "only set args"),
])
def test_suite_validation(tmp_path, monkeypatch, body, match):
    monkeypatch.setattr(suite, "plan", lambda recipe: None)
    with pytest.raises(suite.RecipeError, match=match):
        suite.load_suite(str(_suite(tmp_path, body)))


def test_experiment_resume_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "plan", lambda recipe: None)
    monkeypatch.setenv("DIFFLET_BENCH_ROOT", str(tmp_path))
    s = suite.load_suite("quant_ab")
    with pytest.raises(suite.RecipeError, match="no experiment"):
        suite.open_experiment(s, "e1", resume=True)
    path, manifest = suite.open_experiment(s, "e1", resume=False)
    assert manifest["runs"]["flux/bf16"] == {"status": "pending"}
    with pytest.raises(suite.RecipeError, match="pass --resume"):
        suite.open_experiment(s, "e1", resume=False)
    suite.record(path, manifest, "flux/bf16", {"valid": True, "run_id": "r"}, tmp_path / "r.json")
    _, again = suite.open_experiment(s, "e1", resume=True)
    assert suite.done(again, "flux/bf16") and not suite.done(again, "flux/mxfp8")
    with pytest.raises(suite.RecipeError, match="bad experiment name"):
        suite.experiment_dir("../x")


def _record(run_id, *, args, median, method=1, outputs=("h",), sync=("s",), threads=48,
            host="h1"):
    summary = {"n": 3, "median": median, "mean": median, "min": median, "max": median}
    return {"run_id": run_id, "kind": "result", "method_version": method, "valid": True,
            "recipe": {"name": "flux", "args": args, "variant_args": None, "passthrough": []},
            "metrics": {"dit_step": {"valid": True, "unit": "s", "sync": list(sync),
                                     "summary": dict(summary)},
                        "e2e_warm": {"valid": True, "unit": "s",
                                     "summary": {**summary, "median": median * 100,
                                                 "mean": median * 100}}},
            "launches": [{"command": "generate", "threads": threads, "output": {"sha256": h},
                          "evidence": [{"torch_threads": 1}]} for h in outputs],
            "provenance": {"host_id": host, "git": {"commit": "abc"}}}


def test_compare_refuses_incomparable_runs():
    a = _record("a", args={"steps": 50}, median=0.3)
    with pytest.raises(compare.CompareError, match="methods differ"):
        compare.render(a, _record("b", args={"steps": 50}, median=0.3, method=2))
    noisy = _record("a", args={"steps": 50}, median=0.3, outputs=("h1", "h2"))
    with pytest.raises(compare.CompareError, match="not reproducible"):
        compare.render(noisy, _record("b", args={"steps": 50}, median=0.3))


def test_compare_warns_about_thread_settings_instead_of_refusing():
    a = _record("a", args={"steps": 50}, median=0.3)
    text = compare.render(a, _record("b", args={"steps": 50}, median=0.3, threads=96))
    assert "thread settings differ" in text


def test_compare_reports_what_differs_and_the_ratio():
    a = _record("a", args={"steps": 50}, median=0.300)
    b = _record("b", args={"steps": 50, "quant": "mxfp8"}, median=0.240)
    text = compare.render(a, b)
    assert "--quant: None -> 'mxfp8'" in text
    assert "| dit_step | 0.3 | 0.3 | 0.24 | 0.24 | 0.800 |" in text
    assert "| e2e_warm | 30 | 30 | 24 | 24 | 0.800 |" in text


def test_compare_across_hosts_keeps_only_the_dit_step_ratio():
    a = _record("a", args={"steps": 50}, median=0.300)
    b = _record("b", args={"steps": 50}, median=0.240, host="h2")
    text = compare.render(a, b)
    assert "| dit_step | 0.3 | 0.3 | 0.24 | 0.24 | 0.800 |" in text
    assert "| e2e_warm | 30 | 30 | 24 | 24 |  |" in text


def test_a_suite_measuring_quality_needs_a_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "plan", lambda recipe: None)
    s = suite.load_suite(str(_suite(tmp_path, "metrics: [quality]\nrecipes: [flux]\n")))
    with pytest.raises(suite.RecipeError, match="neither"):
        suite.check_quality(s)
    s.quality_ref = "none"
    suite.check_quality(s)


def test_report_main_table_and_variants(tmp_path):
    base = _record("20261116T000000Z-a-1", args={"steps": 50}, median=0.3)
    newer = _record("20261117T000000Z-a-2", args={"steps": 50}, median=0.29)
    variant = _record("20261117T000000Z-a-3", args={"steps": 50}, median=0.2)
    variant["recipe"].update({"suite": "quant_ab", "variant": "mxfp8",
                              "variant_args": {"quant": "mxfp8"}})
    results.write_json(tmp_path / "trn3" / "flux" / f"{base['run_id']}.json", base)
    results.write_json(tmp_path / "trn3" / "flux" / f"{newer['run_id']}.json", newer)
    results.write_json(tmp_path / "trn3" / "flux@quant_ab.mxfp8" / "v.json", variant)
    written = report.generate(tmp_path)
    summary = (tmp_path / "trn3" / "SUMMARY.md").read_text()
    assert tmp_path / "trn3" / "SUMMARY.md" in written
    assert "290 (n=3)" in summary and "+1 older" in summary  # the latest valid result, in ms
    assert "Suite variants" in summary and "quant_ab.mxfp8" in summary
    assert (tmp_path / "trn3" / "flux" / "REPORT.md").exists()


def test_report_keeps_invalid_results_out_of_the_main_table(tmp_path):
    bad = _record("20261118T000000Z-a-4", args={"steps": 50}, median=0.1)
    bad.update(valid=False, aborted="lane not idle before e2e_cold_1")
    results.write_json(tmp_path / "trn3" / "qwen_image" / f"{bad['run_id']}.json", bad)
    summary = report.render_device("trn3", report._load_device(tmp_path / "trn3"))
    assert "(no valid results)" in summary and "100 (n=3)" not in summary
    assert "without a valid result" in summary and "lane not idle" in summary
