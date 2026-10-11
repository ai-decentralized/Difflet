"""CPU tests of the W6 driver's pure parts: the run plan, the launch judgements, the result
assembly and the Markdown rendering (no device, no subprocess)."""

from __future__ import annotations

import json

import pytest

from benchmark import neuron_cold_warm as cw
from benchmark.adapters import neuron as na
from tests.unit.benchmark import _neuron_results as fake

LAUNCH_PROV = {"head": "3763193", "worktree": "/wt", "status": "?? benchmark/neuron_wan_bench.py",
               "difflet_diff_vs_head": "", "files_sha256": {"benchmark/neuron_wan_bench.py": "aa"},
               "neuronx_cc": ["NeuronX Compiler version 2.28"], "neuron_cc_flags": None,
               "python": "/venv/bin/python"}
PROV = {"head": "3763193", "worktree": "/wt", "launches": 1,
        "status_at_runs": [LAUNCH_PROV["status"]], "difflet_diff_vs_head_at_runs": [""],
        "neuronx_cc": LAUNCH_PROV["neuronx_cc"], "neuron_cc_flags": ["None"],
        "files_sha256_at_runs": {"benchmark/neuron_wan_bench.py": ["aa"]}}


def test_run_provenance_aggregates_the_launch_records():
    other = {**LAUNCH_PROV, "files_sha256": {"benchmark/neuron_wan_bench.py": "aa",
                                             "benchmark/adapters/neuron.py": "bb"}}
    later = {**LAUNCH_PROV, "files_sha256": {"benchmark/neuron_wan_bench.py": "aa",
                                             "benchmark/adapters/neuron.py": "cc"}}
    records = {"g5.0/a": {"provenance": other}, "g5.0/b": {"provenance": later},
               "g5.0/c": {}}
    prov = cw.run_provenance(records)
    assert prov["head"] == "3763193" and prov["launches"] == 2
    assert prov["files_sha256_at_runs"] == {"benchmark/neuron_wan_bench.py": ["aa"],
                                            "benchmark/adapters/neuron.py": ["bb", "cc"]}
    assert prov["difflet_diff_vs_head_at_runs"] == [""]
    two_heads = cw.run_provenance({"x": {"provenance": {**LAUNCH_PROV, "head": "b"}},
                                   "y": {"provenance": LAUNCH_PROV}})
    assert two_heads["head"] == ["3763193", "b"]


# ------------------------------------------------------------------------------ the plan


def test_plan_runs_the_headline_guidance_first_with_one_cold_compile_each():
    plan = cw.build_plan()
    names = [(f"{l.guidance:g}", l.name) for l in plan]
    assert names == [
        ("5", "script-cold"), ("5", "script-warm1"), ("5", "script-warm2"), ("5", "script-warm3"),
        ("5", "cli-w4check"), ("5", "cli-warm1"), ("5", "cli-warm2"), ("5", "cli-warm3"),
        ("5", "cli-truecold"), ("5", "cli-cold"),
        ("1", "script-cold"), ("1", "script-warm1"), ("1", "script-warm2"), ("1", "script-warm3"),
        ("1", "cli-warm1"), ("1", "cli-warm2"), ("1", "cli-warm3"), ("1", "cli-truecold"),
        ("1", "cli-cold")]
    fresh = [(l.guidance, l.name, l.cache) for l in plan if l.fresh_cache]
    assert fresh == [(5.0, "script-cold", "main"), (5.0, "cli-cold", "cli-cold"),
                     (1.0, "script-cold", "main"), (1.0, "cli-cold", "cli-cold")]
    assert [l.name for l in plan if l.drop_page_cache] == ["cli-truecold", "cli-truecold"]
    assert all(l.expect == ("cold" if l.fresh_cache else "warm") for l in plan)
    assert {l.kind for l in plan if l.name == "cli-w4check"} == {"cli-w4"}


def test_plan_without_true_cold_or_w4_check():
    plan = cw.build_plan([5.0], warm_launches=1, cli_warm=1, true_cold=False, w4_guidance=None)
    assert [l.name for l in plan] == ["script-cold", "script-warm1", "cli-warm1", "cli-cold"]


def test_every_row_gets_its_own_cache_root(tmp_path):
    plan = cw.build_plan()
    roots = {(l.guidance, l.cache): cw.cache_root(tmp_path, l) for l in plan}
    assert len(set(roots.values())) == 4
    assert cw.cache_root(tmp_path, plan[0]) == tmp_path / "caches" / "g5.0-main"
    assert cw.run_dir(tmp_path, plan[0]) == tmp_path / "runs" / "g5.0" / "script-cold"


def test_w5_command_is_w4s_parameters(tmp_path):
    cmd = cw.w5_command("/venv/bin/python", output=tmp_path / "o.mp4", cache_dir=tmp_path,
                        work_dir=tmp_path / "w")
    assert cmd[1:5] == ["-m", "difflet.cli.main", "generate", "--backend"]
    pairs = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
    assert pairs["--steps"] == "50" and pairs["--guidance-scale"] == "5"
    assert pairs["--seed"] == "42" and pairs["--tp-degree"] == "4"
    assert pairs["--prompt"] == "a cinematic shot of a red fox running through a snowy forest"
    assert "--exec-mode" not in cmd and "--cp-degree" not in cmd  # W5's line as it was
    assert "--host-vae" in cmd and "--keep-work-dir" in cmd


def test_config_for_overrides_guidance_and_only_smoke_steps(monkeypatch):
    assert cw.config_for(5.0).guidance_scale == 5.0 and cw.config_for(5.0).steps == 50
    monkeypatch.setattr(cw, "STEPS_OVERRIDE", 2)
    assert cw.config_for(1.0).steps == 2


# ------------------------------------------------------------------------- judgements


def _warm_summary(**kwargs):
    return na.summarize_bench_launch(fake.launch_results(**kwargs), steps=50,
                                     guidance_scale=kwargs.get("guidance", 5.0))


RUN_OK = {"exit_code": 0, "timed_out": False}


def _by_name(checks):
    return {c["name"]: c for c in checks}


def test_judge_bench_warm_launch_passes():
    launch = cw.Launch("script-warm1", "bench", 5.0, "main", "warm")
    checks = _by_name(cw.judge_bench(launch, _warm_summary(), RUN_OK, {}, fake.LATENTS))
    assert all(c["ok"] for c in checks.values()), [c for c in checks.values() if not c["ok"]]
    assert checks["counter_rule_warm"]["hard"] and checks["no_compile_in_generates"]["hard"]
    assert not checks["step_series_it1"]["hard"]
    assert not checks["resident_medians_within_2pct"]["hard"]
    assert checks["latents_equal_w4"]["ok"] and not checks["latents_equal_w4"]["hard"]


def test_judge_bench_hard_failures():
    launch = cw.Launch("script-warm2", "bench", 5.0, "main", "warm")
    results = fake.launch_results()
    results[1]["iterations"][0]["fallbacks"] = ["aten::index"]
    summary = na.summarize_bench_launch(results, steps=50, guidance_scale=5.0)
    checks = _by_name(cw.judge_bench(launch, summary, RUN_OK, {"aa/bb.neff": 10}, "other"))
    failed = {name for name, c in checks.items() if not c["ok"] and c["hard"]}
    assert failed == {"fallbacks", "counter_rule_warm", "latents_equal_to_cold_launch"}
    exited = cw.judge_bench(launch, None, {"exit_code": 1, "timed_out": False}, {}, None)
    assert [(c["name"], c["ok"]) for c in exited] == [("exit", False)]


def test_judge_bench_cold_launch_may_compile_in_its_first_generate():
    launch = cw.Launch("script-cold", "bench", 1.0, "main", "cold", fresh_cache=True)
    results = fake.cold_results(guidance=1.0)
    results[0]["iterations"][0]["counters"]["TotalCompilations"] = 3
    summary = na.summarize_bench_launch(results, steps=50, guidance_scale=1.0)
    checks = _by_name(cw.judge_bench(launch, summary, RUN_OK, {f"{i}.neff": 10 for i in range(47)},
                                     None))
    assert checks["counter_rule_cold"]["ok"] and checks["counter_rule_cold"]["hard"]
    assert not checks["no_compile_in_generates"]["ok"]
    assert not checks["no_compile_in_generates"]["hard"]
    assert "latents_equal_w4" not in checks  # guidance 1.0
    assert "latents_equal_to_cold_launch" not in checks


def _cli(**overrides):
    parsed = na.parse_cli_log(
        "".join(f"[wan] neuron rank {r}/4: exec_mode=compile compiled_blocks=40 "
                "unwarmed_shapes=[] fallbacks: load=[] generate=[] load+warmup 30.0s, "
                "stage 90.0s\n" for r in range(4))
        + "[wan] latents saved to /w/latents.pt\n[wan] video saved to /o.mp4\n")
    cli = {"launch": {"exit_code": 0, "timed_out": False}, "parsed": parsed,
           "checks": na.cli_checks(parsed),
           "inspect": {"tensor_sha256": cw.W4_EXPECTED["latents_tensor_sha256"],
                       "file_sha256": cw.W4_EXPECTED["latents_file_sha256"],
                       "mp4": {"sha256": cw.W4_EXPECTED["mp4_sha256"]}}}
    cli.update(overrides)
    return cli


def test_judge_cli_warm_file_rule_and_w4_hashes():
    w4 = cw.Launch("cli-w4check", "cli-w4", 5.0, "main", "warm")
    checks = _by_name(cw.judge_cli(w4, _cli(), {}, 0, fake.LATENTS))
    assert all(c["ok"] for c in checks.values())
    assert {"w4_latents_file", "w4_latents_tensor", "w4_mp4"} <= set(checks)
    assert checks["latents_equal_to_script"]["hard"]

    wrong = _cli(inspect={"tensor_sha256": "x", "file_sha256": "y", "mp4": {"sha256": "z"}})
    failed = {n for n, c in _by_name(cw.judge_cli(w4, wrong, {}, 0, fake.LATENTS)).items()
              if not c["ok"] and c["hard"]}
    assert failed == {"w4_latents_file", "w4_latents_tensor", "w4_mp4",
                      "latents_equal_to_script"}

    warm = cw.Launch("cli-warm1", "cli", 1.0, "main", "warm")
    grew = _by_name(cw.judge_cli(warm, _cli(), {"a.neff": 1}, 0, None))
    assert not grew["file_rule_warm"]["ok"] and grew["file_rule_warm"]["hard"]
    assert not _by_name(cw.judge_cli(warm, _cli(), {}, 3, None))["file_rule_warm"]["ok"]
    other = _by_name(cw.judge_cli(warm, _cli(), {}, 0, "different"))
    assert not other["latents_equal_to_script"]["ok"]
    assert not other["latents_equal_to_script"]["hard"]  # a flag outside the W4 check

    cold = cw.Launch("cli-cold", "cli", 1.0, "cli-cold", "cold", fresh_cache=True)
    assert _by_name(cw.judge_cli(cold, _cli(), {"a.neff": 1}, 132, None))["file_rule_cold"]["ok"]
    assert not _by_name(cw.judge_cli(cold, _cli(), {}, 0, None))["file_rule_cold"]["ok"]


# --------------------------------------------------------------------------- assembly


def _records(g: float, *, with_cli: bool = True) -> dict:
    key = cw.gkey(g)
    cold = na.summarize_bench_launch(fake.cold_results(guidance=g), steps=50, guidance_scale=g)
    neffs = {f"{i:02d}/x.neff": (5_050_000 if i < 4 else 20_000) for i in range(47)}
    records = {f"{key}/script-cold": fake.launch_record(
        cold, name="script-cold", guidance=g, expect="cold", wall=500.0, new_neffs=neffs)}
    for i, e2e_shift in enumerate((0.0, 2.0, 4.0), start=1):
        results = fake.launch_results(guidance=g, step=1.07 + 0.001 * i)
        for r in results:
            r["iterations"][0]["t_mp4_saved_s"] = (r["iterations"][0].get("t_mp4_saved_s")
                                                   or 149.0) + e2e_shift
        summary = na.summarize_bench_launch(results, steps=50, guidance_scale=g)
        records[f"{key}/script-warm{i}"] = fake.launch_record(
            summary, name=f"script-warm{i}", guidance=g, wall=420.0 + i)
    if with_cli:
        for i, wall in enumerate((180.0, 182.0, 184.0), start=1):
            records[f"{key}/cli-warm{i}"] = fake.cli_record(name=f"cli-warm{i}", guidance=g,
                                                            wall=wall)
        records[f"{key}/cli-cold"] = fake.cli_record(
            name="cli-cold", guidance=g, wall=260.0, expect="cold",
            new_neffs={f"{i}.neff": 1 for i in range(47)})
        records[f"{key}/cli-truecold"] = fake.cli_record(
            name="cli-truecold", guidance=g, wall=420.0,
            drop={"ok": True, "before": {"Cached_gib": 80.0}, "after": {"Cached_gib": 1.2}})
    return records


def test_guidance_numbers_follow_the_definitions():
    records = _records(5.0)
    n = cw.guidance_numbers(records, 5.0)
    assert n["compile_wall_s"] == pytest.approx(78.0 - 13.0)
    assert n["compiling_rank"]["rank"] == 0
    assert n["cold_new_neffs"] == 47 and n["cold_block_files"] == 4
    assert n["e2e_warm"]["n"] == 3
    assert n["e2e_warm"]["samples"] == pytest.approx([149.0, 151.0, 153.0])
    assert n["it1"]["step"]["median"] == pytest.approx(1.071)  # the first warm launch
    assert n["resident_step"]["n"] == 6 * 49
    assert n["cli_warm_wall"]["mean"] == pytest.approx(182.0)
    assert n["peak_gib"] == pytest.approx(8.03)
    assert cw.guidance_numbers({}, 5.0) is None
    missing_warm = {k: v for k, v in records.items() if "script-warm" not in k}
    assert cw.guidance_numbers(missing_warm, 5.0) is None


def test_bench_result_fills_the_harness_schema_and_the_table():
    records = _records(5.0)
    n = cw.guidance_numbers(records, 5.0)
    doc = cw.bench_result(n, records=records, device="trn2.3xlarge", toolchain={"torch": "2"},
                          prov=PROV)
    assert doc["backend"] == "neuron" and doc["guidance_scale"] == 5.0
    assert doc["config_slug"] == "wan_2_1_neuron" and doc["steps"] == 50
    assert doc["compile_seconds"] == pytest.approx(65.0)
    assert doc["e2e_cold_seconds"] == pytest.approx(149.0)
    assert doc["e2e_warm"]["n"] == 3 and doc["step_latency"]["n"] == 49
    assert doc["load_seconds"] == pytest.approx(3.6)
    assert doc["step_basis"].startswith("real loop, neuron synchronize both sides")
    assert doc["throughput"]["DiT steps/s"] == pytest.approx(1 / 1.071)
    assert doc["stage_seconds"]["text_encoder_load_s"] == pytest.approx(0.7)
    assert doc["output"]["shape"] == [1, 3, 9, 480, 832]
    assert all(isinstance(v, float) for v in doc["compile_breakdown"].values())
    assert doc["status"] == "ok"
    metrics = [row[0] for row in doc["neuron"]["table"]]
    for row in ("compile wall (s)", "NEFF compile (s) / NEFFs", "counters per rank", "load (s)",
                "prepare_runtime / load() total (s)",
                "text encoder load (s)", "e2e (s), script", "e2e (s), script, latent output",
                "e2e (s), CLI", "e2e (s), CLI, true cold", "text encode / denoise / VAE decode (s)",
                "DiT step median (ms)", "DiT step rank-max median (ms)",
                "DiT step mean / p90 / min / max (ms)", "DiT forward (ms)",
                "throughput (steps/s)", "peak device mem (GiB/rank)",
                "fallbacks / unwarmed shapes", "throttle", "output"):
        assert row in metrics, row
    assert all(len(row) == 6 for row in doc["neuron"]["table"])
    assert all(str(cell).strip() for row in doc["neuron"]["table"] for cell in row)
    json.dumps(doc)  # JSON-serialisable


def test_true_cold_row_says_when_it_was_skipped():
    records = _records(1.0)
    del records["g1.0/cli-truecold"]
    n = cw.guidance_numbers(records, 1.0)
    row = next(r for r in cw.w6_rows(n) if r[0] == "e2e (s), CLI, true cold")
    assert row[2].startswith("skipped")


def test_render_markdown_puts_guidance_five_first():
    records = {**_records(1.0), **_records(5.0)}
    results = {}
    for g in (5.0, 1.0):
        n = cw.guidance_numbers(records, g)
        results[f"{g:.1f}"] = cw.bench_result(n, records=records, device="trn2.3xlarge",
                                              toolchain={}, prov=PROV)
    md = cw.render_markdown(results, plan_note="Run order: guidance 5, 1.")
    assert md.index("## Guidance 5.0") < md.index("## Guidance 1.0")
    headline = md[md.index("## Headline"):md.index("## Guidance 5.0")]
    assert headline.index("| 5.0 |") < headline.index("| 1.0 |")
    assert "| e2e (s), CLI, true cold |" in md and "Run order: guidance 5, 1." in md
    appendix = md[md.index("## Appendix: generic report, guidance 5.0"):]
    assert "\n## Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers" in appendix  # demoted H1
    assert "## Reproduction\n" not in appendix and "### Reproduction" not in appendix


def test_demote_headings_and_drop_section():
    text = ("# A\n## B\nx\n```bash\n# not a heading\n```\n## Reproduction\ny\n```bash\n"
            "# a comment in the dropped block\n```\n### Sub\nz\n## C\n")
    assert cw.demote_headings("# A\n```\n# c\n```\n## B") == "## A\n```\n# c\n```\n### B"
    assert cw.drop_section(text, "## Reproduction") == (
        "# A\n## B\nx\n```bash\n# not a heading\n```\n## C")


def test_load_records_rejudges_from_the_rank_results_in_plan_order(tmp_path):
    def write(launch, record, results=None):
        out = cw.run_dir(tmp_path, launch)
        out.mkdir(parents=True)
        for r in results or []:
            (out / f"result-rank{r['rank']}.json").write_text(json.dumps(r))
        record["launch"] = cw.dataclasses.asdict(launch)
        (out / "record.json").write_text(json.dumps(record))

    cold = cw.Launch("script-cold", "bench", 5.0, "main", "cold", fresh_cache=True)
    warm = cw.Launch("script-warm1", "bench", 5.0, "main", "warm")
    cli = cw.Launch("cli-warm1", "cli", 5.0, "main", "warm")
    stale = {"summary": None, "checks": [], "ok": False,
             "run": {"exit_code": 0, "timed_out": False}}
    write(cli, {**fake.cli_record(name="cli-warm1", guidance=5.0, wall=180.0),
                "cli": {**_cli(), "launch": {"exit_code": 0, "timed_out": False}},
                "state_before": {"hlo_files": 132}, "state_after": {"hlo_files": 132}})
    write(warm, {**stale, "new_neffs": {}}, fake.launch_results())
    write(cold, {**stale, "new_neffs": {f"{i}.neff": 1 for i in range(47)}},
          fake.cold_results())
    records = cw.load_records(tmp_path, rejudge=True)
    assert list(records) == ["g5.0/script-cold", "g5.0/script-warm1", "g5.0/cli-warm1"]
    assert all(rec["ok"] for rec in records.values())
    warm_checks = _by_name(records["g5.0/script-warm1"]["checks"])
    assert warm_checks["latents_equal_to_cold_launch"]["ok"]
    assert records["g5.0/script-warm1"]["summary"]["iterations"][0]["step"]["n"] == 49
    assert not cw.load_records(tmp_path)["g5.0/script-warm1"]["ok"]  # as stored


def test_check_lines_show_the_rule_and_every_non_pass():
    checks = [
        {"launch": "script-warm1", "name": "exit", "ok": True, "detail": "exit 0", "hard": True},
        {"launch": "script-warm1", "name": "counter_rule_warm", "ok": True,
         "detail": "r0 0/39/7459", "hard": True},
        {"launch": "script-warm1", "name": "step_series_it2", "ok": False,
         "detail": "p90/median 1.2", "hard": False},
        {"launch": "cli-warm1", "name": "file_rule_warm", "ok": False, "detail": "new .neff 1",
         "hard": True},
    ]
    lines = cw.check_lines(checks)
    assert lines[0] == ("- `script-warm1`: 2/3 pass; counter_rule_warm: r0 0/39/7459; "
                        "**flag** step_series_it2: p90/median 1.2")
    assert lines[1] == ("- `cli-warm1`: 0/1 pass; file_rule_warm: new .neff 1; **FAIL** "
                        "file_rule_warm: new .neff 1")


def test_table_md_escapes_pipes():
    md = cw.table_md([["a|b", "c", "d", "e", "f", "g"]])
    assert "a\\|b" in md and md.count("\n") == 2
