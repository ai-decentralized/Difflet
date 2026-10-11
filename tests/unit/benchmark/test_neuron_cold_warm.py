"""CPU tests of the W6 driver's pure parts: the run plan, the launch judgements, the result
assembly and the Markdown rendering (no device; subprocesses only where stated)."""

from __future__ import annotations

import json

import pytest

from benchmark import neuron_cold_warm as cw
from benchmark import neuron_report as nr
from benchmark.adapters import neuron as na
from tests.unit.benchmark import _neuron_results as fake

SCRIPT_HEAD = "3763193977f4a508f72f713444d100f65fa88b08"
CLI_HEAD = "386a03181a10a1106fe00d1cf0b29b21e605ac99"


def test_run_provenance_aggregates_the_launch_records():
    other = {**fake.provenance(), "files_sha256": {"benchmark/neuron_wan_bench.py": "aa",
                                                   "benchmark/adapters/neuron.py": "bb"}}
    later = {**fake.provenance(), "files_sha256": {"benchmark/neuron_wan_bench.py": "aa",
                                                   "benchmark/adapters/neuron.py": "cc"}}
    records = {"g5.0/a": {"provenance": other}, "g5.0/b": {"provenance": later},
               "g5.0/c": {}}
    prov = cw.run_provenance(records)
    assert prov["head"] == [SCRIPT_HEAD] and prov["launches"] == 2  # always a list (M-14)
    assert prov["files_sha256_at_runs"] == {"benchmark/neuron_wan_bench.py": ["aa"],
                                            "benchmark/adapters/neuron.py": ["bb", "cc"]}
    assert prov["loaded_sha256_at_runs"] == {"benchmark/neuron_cold_warm.py": ["bb"]}
    assert prov["difflet_diff_vs_head_at_runs"] == [""]
    assert prov["difflet_trees"] == [fake.TREE]
    two_heads = cw.run_provenance({"x": {"provenance": fake.provenance(CLI_HEAD)},
                                   "y": {"provenance": fake.provenance()}})
    assert two_heads["head"] == [SCRIPT_HEAD, CLI_HEAD]


# ------------------------------------------------------------------------------ the plan


def test_plan_runs_the_headline_guidance_first_with_one_cold_compile_each():
    plan = cw.build_plan()
    names = [(f"{l.guidance:g}", l.name) for l in plan]
    assert names == [
        ("5", "script-cold"), ("5", "script-warm1"), ("5", "script-warm2"), ("5", "script-warm3"),
        ("5", "cli-cold"), ("5", "cli-w4check"), ("5", "cli-warm1"), ("5", "cli-warm2"),
        ("5", "cli-warm3"), ("5", "cli-truecold"),
        ("1", "script-cold"), ("1", "script-warm1"), ("1", "script-warm2"), ("1", "script-warm3"),
        ("1", "cli-cold"), ("1", "cli-warm1"), ("1", "cli-warm2"), ("1", "cli-warm3"),
        ("1", "cli-truecold")]
    fresh = [(l.guidance, l.name, l.cache) for l in plan if l.fresh_cache]
    assert fresh == [(5.0, "script-cold", "main"), (5.0, "cli-cold", "cli-default"),
                     (1.0, "script-cold", "main"), (1.0, "cli-cold", "cli-default")]
    # D82: every CLI launch on the CLI rows' own cache
    assert {l.cache for l in plan if l.kind != "bench"} == {"cli-default"}
    assert [l.name for l in plan if l.drop_page_cache] == ["cli-truecold", "cli-truecold"]
    assert all(l.expect == ("cold" if l.fresh_cache else "warm") for l in plan)
    assert {l.kind for l in plan if l.name == "cli-w4check"} == {"cli-w4"}


def test_plan_without_true_cold_or_w4_check():
    plan = cw.build_plan([5.0], warm_launches=1, cli_warm=1, true_cold=False, w4_guidance=None)
    assert [l.name for l in plan] == ["script-cold", "script-warm1", "cli-cold", "cli-warm1"]


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


def test_output_paths_keep_the_repo_files_for_a_completed_job(tmp_path):
    args = cw.argparse.Namespace(out_json=None, out_md=None)
    repo = na.REPO_ROOT / "benchmark" / "trn2"
    assert cw.output_paths(args, tmp_path, final=True) == (repo / "wan_2_1_neuron.json",
                                                            repo / "wan_2_1_neuron.md")
    assert cw.output_paths(args, tmp_path, final=False) == (  # stopped / --assemble-only
        tmp_path / "results" / "wan_2_1_neuron.json", tmp_path / "results" / "wan_2_1_neuron.md")
    explicit = cw.argparse.Namespace(out_json=tmp_path / "a.json", out_md=tmp_path / "a.md")
    assert cw.output_paths(explicit, tmp_path, final=False) == (tmp_path / "a.json",
                                                                 tmp_path / "a.md")


# -------------------------------------------------------------------------- host checks


def test_wait_cores_free_polls_until_neuron_ls_and_ps_are_clear(monkeypatch):
    idle = json.dumps([{"neuron_device": 0, "neuron_processes": []}])
    busy = json.dumps([{"neuron_device": 0, "neuron_processes": [{"pid": 7}]}])
    outputs = {"neuron-ls": [busy, idle, idle],
               "ps": ["  PID COMMAND\n  9 python -m torch.distributed.run x\n",
                      "  PID COMMAND\n  9 python -m torch.distributed.run x\n",
                      "  PID COMMAND\n  10 grep -E torchrun|neuron_wan_bench\n"]}
    calls = []

    def fake_run_text(cmd, timeout=60.0):
        kind = "neuron-ls" if cmd[0].endswith("neuron-ls") else "ps"
        calls.append(kind)
        return outputs[kind].pop(0)

    monkeypatch.setattr(cw, "_run_text", fake_run_text)
    monkeypatch.setattr(cw.time, "sleep", lambda s: None)
    assert cw.wait_cores_free(timeout=60) == {"neuron_ls_processes": [], "ps_busy": []}
    assert calls == ["neuron-ls", "ps"] * 3  # busy device, busy torchrun, then clear

    monkeypatch.setattr(cw, "_run_text", lambda cmd, timeout=60.0: busy if cmd[0].endswith(
        "neuron-ls") else "  PID COMMAND\n")
    with pytest.raises(RuntimeError, match="cores not free"):
        cw.wait_cores_free(timeout=0)


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
    w4 = cw.Launch("cli-w4check", "cli-w4", 5.0, "cli-default", "warm")
    checks = _by_name(cw.judge_cli(w4, _cli(), {}, 0, fake.LATENTS))
    assert all(c["ok"] for c in checks.values())
    assert {"w4_latents_file", "w4_latents_tensor", "w4_mp4"} <= set(checks)
    assert checks["latents_equal_to_script"]["hard"]

    wrong = _cli(inspect={"tensor_sha256": "x", "file_sha256": "y", "mp4": {"sha256": "z"}})
    failed = {n for n, c in _by_name(cw.judge_cli(w4, wrong, {}, 0, fake.LATENTS)).items()
              if not c["ok"] and c["hard"]}
    assert failed == {"w4_latents_file", "w4_latents_tensor", "w4_mp4",
                      "latents_equal_to_script"}

    warm = cw.Launch("cli-warm1", "cli", 1.0, "cli-default", "warm")
    grew = _by_name(cw.judge_cli(warm, _cli(), {"a.neff": 1}, 0, None))
    assert not grew["file_rule_warm"]["ok"] and grew["file_rule_warm"]["hard"]
    assert not _by_name(cw.judge_cli(warm, _cli(), {}, 3, None))["file_rule_warm"]["ok"]
    other = _by_name(cw.judge_cli(warm, _cli(), {}, 0, "different"))
    assert not other["latents_equal_to_script"]["ok"]
    assert not other["latents_equal_to_script"]["hard"]  # a flag outside the W4 check

    cold = cw.Launch("cli-cold", "cli", 1.0, "cli-default", "cold", fresh_cache=True)
    assert _by_name(cw.judge_cli(cold, _cli(), {"a.neff": 1}, 132, None))["file_rule_cold"]["ok"]
    assert not _by_name(cw.judge_cli(cold, _cli(), {}, 0, None))["file_rule_cold"]["ok"]


# ---------------------------------------------------------------------------- launches


class _RaisingAdapter:
    python = "/venv/bin/python"

    def run_cli_generate(self, *args, **kwargs):
        raise ValueError("no result-rank2.json")

    run_bench = run_cli_generate


def _no_host(monkeypatch):
    monkeypatch.setattr(cw, "wait_cores_free", lambda: {"neuron_ls_processes": []})
    monkeypatch.setattr(cw, "provenance", lambda: fake.provenance())
    monkeypatch.setattr(cw, "host_state", lambda root: {"neff": {}, "hlo_files": 0})


def test_run_launch_records_an_exception_and_keeps_the_previous_attempt(tmp_path, monkeypatch):
    _no_host(monkeypatch)
    launch = cw.Launch("cli-warm1", "cli", 5.0, "cli-default", "warm")
    out = cw.run_dir(tmp_path, launch)
    out.mkdir(parents=True)
    (out / "record.json").write_text(json.dumps({"ok": False, "evidence": "first attempt"}))
    rec = cw.run_launch(launch, record_dir=tmp_path, adapter=_RaisingAdapter(), iters=3,
                        records={}, host=fake.HOST)
    assert not rec["ok"] and rec["checks"] == [{"name": "exception", "ok": False, "hard": True,
                                                "detail": "ValueError: no result-rank2.json"}]
    assert json.loads((out / "record.json").read_text())["checks"][0]["name"] == "exception"
    kept = [p for p in out.parent.iterdir() if p.name.startswith("cli-warm1.failed-")]
    assert len(kept) == 1 and rec["previous_attempt"] == kept[0].name  # M-5: evidence kept
    assert json.loads((kept[0] / "record.json").read_text())["evidence"] == "first attempt"
    assert rec["host"] == fake.HOST
    # load_records skips a set-aside attempt
    assert list(cw.load_records(tmp_path)) == ["g5.0/cli-warm1"]


def test_a_failure_before_the_launch_is_a_failed_record(tmp_path, monkeypatch):
    _no_host(monkeypatch)

    def busy():
        raise RuntimeError("cores not free: neuron-ls [{'pid': 7}], ps []")

    monkeypatch.setattr(cw, "wait_cores_free", busy)
    launch = cw.Launch("cli-cold", "cli", 5.0, "cli-default", "cold", fresh_cache=True)
    rec = cw.run_launch(launch, record_dir=tmp_path, adapter=_RaisingAdapter(), iters=3,
                        records={}, host=fake.HOST)
    assert not rec["ok"]
    assert rec["checks"][0]["detail"].startswith("RuntimeError: cores not free")
    assert json.loads((cw.run_dir(tmp_path, launch) / "record.json").read_text())["ok"] is False


def _write_bench_launch(root, name, guidance, results, *, expect="warm", start=990.0,
                        new_neffs=None):
    launch = cw.Launch(name, "bench", guidance, "main", expect, fresh_cache=expect == "cold")
    out = cw.run_dir(root, launch)
    out.mkdir(parents=True)
    for r in results:
        (out / f"result-rank{r['rank']}.json").write_text(json.dumps(r))
    summary = na.summarize_bench_launch(results, steps=50, guidance_scale=guidance)
    record = fake.launch_record(summary, name=name, guidance=guidance, expect=expect,
                                start_epoch=start, new_neffs=new_neffs)
    record["launch"] = cw.dataclasses.asdict(launch)
    (out / "record.json").write_text(json.dumps(record))


def test_main_stops_on_an_interrupt_and_assembles_into_results(tmp_path, monkeypatch):
    _write_bench_launch(tmp_path, "script-cold", 5.0, fake.cold_results(), expect="cold",
                        new_neffs={f"{i}.neff": 1 for i in range(47)})
    for i in (1, 2, 3):
        _write_bench_launch(tmp_path, f"script-warm{i}", 5.0, fake.launch_results(),
                            start=1000.0 + i)
    launched = []

    def interrupted(launch, **kwargs):
        launched.append(launch.name)
        raise KeyboardInterrupt("signal 15")  # run_logged has stopped the launch by now

    monkeypatch.setattr(cw, "run_launch", interrupted)
    monkeypatch.setattr(cw, "sudo_available", lambda: False)
    monkeypatch.setattr(cw, "host_info", lambda adapter: dict(fake.HOST))
    repo = [na.REPO_ROOT / "benchmark" / "trn2" / f"wan_2_1_neuron.{ext}" for ext in ("json", "md")]
    before = [(p.stat().st_mtime_ns, p.read_bytes()) for p in repo]
    handler = cw.signal.getsignal(cw.signal.SIGTERM)
    try:
        code = cw.main(["--record-dir", str(tmp_path), "--skip-existing", "--guidance", "5.0",
                        "--true-cold", "no"])
        assert cw.signal.getsignal(cw.signal.SIGTERM) is cw._interrupt  # SIGTERM = Ctrl-C
    finally:
        cw.signal.signal(cw.signal.SIGTERM, handler)
    assert code == 130 and launched == ["cli-cold"]  # the four script launches were reused
    doc = json.loads((tmp_path / "results" / "wan_2_1_neuron.json").read_text())
    assert list(doc["results_by_guidance"]) == ["5.0"]
    assert (tmp_path / "results" / "wan_2_1_neuron.md").is_file()
    assert [(p.stat().st_mtime_ns, p.read_bytes()) for p in repo] == before  # untouched


def test_bench_script_note_says_how_the_tree_relates_to_the_runs():
    ran, now = ("adb718622beb1a63b7e836caa4bc1953dc158602e7376052c5d14411d587ae7a",
                "36751f0ed5cad9e22c2503da3e7622fe2136a04f8bbd58c1c878adbb1792e89b")
    prov = {"files_sha256_at_runs": {cw.BENCH_SCRIPT: [ran]}, "launches": 19}
    same = cw.bench_script_note(prov, current=ran)
    assert same.endswith("(adb718622beb…, byte-identical); the copy in this tree is the same file.")
    known = cw.bench_script_note(prov, current=now)
    assert "the copy in this tree (36751f0ed5ca…) differs from it: only the W6 review's M-10" in known
    assert "behaviour and the recorded tags are the same" in known
    assert "not described here" in cw.bench_script_note(prov, current="f" * 64)


def test_run_order_is_the_order_the_launches_started():
    def rec(name, start, head, cache):
        return {"launch": {"name": name, "cache": cache},
                "run": {"start_epoch": start, "end_epoch": start + 100},
                "provenance": {"head": head}}

    base = 1791712800.0  # 2026-10-11 10:00 UTC
    records = {"g5.0/script-cold": rec("script-cold", base, SCRIPT_HEAD, "main"),
               "g5.0/cli-cold": rec("cli-cold", base + 7200, CLI_HEAD, "cli-default"),
               "g5.0/cli-truecold": rec("cli-truecold", base + 600, SCRIPT_HEAD, "main"),
               "g1.0/script-cold": rec("script-cold", base + 1200, SCRIPT_HEAD, "main")}
    assert cw.plan_note_of(records) == (
        "Run order (by start time): g=5: script-cold, cli-truecold (10:00-10:11 UTC, HEAD "
        "3763193, cache g5.0-main); then cli-cold (12:00-12:01 UTC, HEAD 386a031, cache "
        "g5.0-cli-default) | g=1: script-cold (10:20-10:21 UTC, HEAD 3763193, cache g1.0-main).")


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
        records[f"{key}/cli-cold"] = fake.cli_record(
            name="cli-cold", guidance=g, wall=216.0, expect="cold",
            new_neffs={f"{i}.neff": 1 for i in range(47)})
        for i, wall in enumerate((159.0, 160.0, 161.0), start=1):
            records[f"{key}/cli-warm{i}"] = fake.cli_record(name=f"cli-warm{i}", guidance=g,
                                                            wall=wall)
        records[f"{key}/cli-truecold"] = fake.cli_record(
            name="cli-truecold", guidance=g, wall=780.0, omp="3", driver_threads=3,
            head=SCRIPT_HEAD,
            drop={"ok": True, "before": {"Cached_gib": 80.0}, "after": {"Cached_gib": 1.2}})
    return records


def _result(records, g):
    n = cw.guidance_numbers(records, g)
    return cw.bench_result(n, records=records, device="trn2.3xlarge", toolchain={"torch": "2"},
                           prov=cw.run_provenance(records))


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
    assert n["cli_warm_wall"]["mean"] == pytest.approx(160.0)
    assert n["peak_gib"] == pytest.approx(8.03)
    assert n["threads"] == {"bench_omp": ["3"],
                            "cli": {"omp": ["unset"], "driver": [6]},
                            "true_cold": {"omp": ["3"], "driver": [3]}}
    assert cw.guidance_numbers({}, 5.0) is None
    missing_warm = {k: v for k, v in records.items() if "script-warm" not in k}
    assert cw.guidance_numbers(missing_warm, 5.0) is None


def test_warm_launches_keep_their_names(monkeypatch):
    records = _records(5.0)
    del records["g5.0/script-warm2"]
    n = cw.guidance_numbers(records, 5.0)
    assert n["warm_names"] == ["script-warm1", "script-warm3"]
    counters_row = next(r for r in nr.w6_rows(n) if r[0] == "counters per rank")
    assert "warm1:" in counters_row[3] and "warm3:" in counters_row[3]
    assert "warm2:" not in counters_row[3]
    # the headline series needs script-warm1 itself, never a stand-in (M-9)
    records = _records(5.0)
    records["g5.0/script-warm1"]["ok"] = False
    assert cw.guidance_numbers(records, 5.0) is None


def test_bench_result_fills_the_harness_schema_and_the_table():
    records = _records(5.0)
    doc = _result(records, 5.0)
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


def test_cli_rows_state_the_recorded_threads_and_the_true_cold_caveat():
    rows = {r[0]: r for r in cw.w6_rows(cw.guidance_numbers(_records(5.0), 5.0))}
    cli = rows["e2e (s), CLI"]
    assert "OMP_NUM_THREADS unset in the launch env" in cli[1]
    assert "decode at 6 threads" in cli[1] and "=3" not in cli[1]
    true_cold = rows["e2e (s), CLI, true cold"][2]
    assert "caveat: driver VAE decode at 3 threads" in true_cold
    assert "CLI warm/cold rows at the default (6 threads)" in true_cold
    assert "decode at 6 threads" in rows["CLI start → latents / → mp4 (s)"][1]
    assert rows["CLI start → latents / → mp4 (s)"][3].startswith("98.0 / 60.0")  # recorded


def test_notes_name_the_commit_of_each_row_kind_and_the_flags():
    records = _records(5.0)
    notes = _result(records, 5.0)["notes"]
    assert notes[0].startswith(f"HEAD at the launches: 3763193 (script, true-cold CLI rows); "
                               f"386a031 (CLI rows), all in /wt; difflet/ identical at every "
                               f"launch (tree {fake.TREE[:12]}, no uncommitted change)")
    assert any("NEURON_CC_FLAGS at the launches: None" in note for note in notes)
    threads = next(note for note in notes if note.startswith("Threads (as recorded)"))
    assert "OMP_NUM_THREADS in the launch env unset" in threads
    assert "decodes the VAE at 6 threads" in threads
    assert "true-cold launch: OMP_NUM_THREADS 3, driver decode at 3 threads" in threads


def test_status_is_flagged_by_a_soft_check_and_partial_by_a_hard_one():
    records = _records(5.0)
    assert cw.status_of(records, 5.0) == "ok"
    records["g5.0/script-warm2"]["checks"].append(
        {"name": "step_series_it1", "ok": False, "detail": "p90/median 1.2", "hard": False})
    assert cw.status_of(records, 5.0) == "flagged"  # M-1: shown, not fatal
    assert _result(records, 5.0)["status"] == "flagged"
    records["g5.0/cli-warm1"]["checks"].append(
        {"name": "file_rule_warm", "ok": False, "detail": "new .neff 1", "hard": True})
    assert cw.status_of(records, 5.0) == "partial"


def test_true_cold_row_says_when_it_was_skipped():
    records = _records(1.0)
    del records["g1.0/cli-truecold"]
    n = cw.guidance_numbers(records, 1.0)
    row = next(r for r in nr.w6_rows(n) if r[0] == "e2e (s), CLI, true cold")
    assert row[2].startswith("skipped")
    assert "true-cold row was not run" in cw.plan_note_of(records)


def test_render_markdown_puts_guidance_five_first():
    records = {**_records(1.0), **_records(5.0)}
    results = {f"{g:.1f}": _result(records, g) for g in (5.0, 1.0)}
    md = nr.render_markdown(results, plan_note="Run order: guidance 5, 1.")
    assert md.index("## Guidance 5.0") < md.index("## Guidance 1.0")
    headline = md[md.index("## Headline"):md.index("## Guidance 5.0")]
    assert headline.index("| 5.0 |") < headline.index("| 1.0 |")
    assert "| e2e (s), CLI, true cold |" in md and "Run order: guidance 5, 1." in md
    assert "**Commit(s):** `3763193977f4`, `386a03181a10` in `/wt` (difflet/ identical" in md
    appendix = md[md.index("## Appendix: generic report, guidance 5.0"):]
    assert "\n## Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers" in appendix  # demoted H1
    assert "## Reproduction\n" not in appendix and "### Reproduction" not in appendix
    # M-11: the AoT-only statements are gone from the generic report
    assert "(page cache dropped)" not in appendix and "> Cold vs warm:" not in appendix
    assert "cold start** (NEFF cache empty; page cache hot)" in appendix


def test_neutralize_aot_wording():
    text = ("| **e2e generate — cold start** (page cache dropped) | **1 s** |\n\n"
            "> Cold vs warm: **a → b**. the gap is the one-time cold disk read.\n\nnext")
    assert nr.neutralize_aot_wording(text) == (
        "| **e2e generate — cold start** (NEFF cache empty; page cache hot) | **1 s** |\n\nnext")


def _write_records(root, records):
    for key, rec in records.items():
        out = root / "runs" / key
        out.mkdir(parents=True)
        (out / "record.json").write_text(json.dumps(rec))


def test_assemble_is_deterministic_and_reads_only_the_records(tmp_path):
    records = {**_records(5.0), **_records(1.0)}
    _write_records(tmp_path, records)
    first = tmp_path / "a.json", tmp_path / "a.md"
    again = tmp_path / "b.json", tmp_path / "b.md"
    for out_json, out_md in (first, again):
        # adapter=None: the host facts come from the records, nothing is probed (M-2)
        cw.assemble(cw.load_records(tmp_path), adapter=None, out_json=out_json, out_md=out_md)
    assert first[0].read_bytes() == again[0].read_bytes()
    assert first[1].read_bytes() == again[1].read_bytes()
    doc = json.loads(first[0].read_text())
    assert doc["device"] == fake.HOST["device"] and doc["toolchain"] == fake.HOST["toolchain"]
    assert list(doc["results_by_guidance"]) == ["5.0", "1.0"]
    assert doc["plan_note"].startswith("Run order (by start time): g=")
    assert " g=5: " in doc["plan_note"] or doc["plan_note"].startswith(
        "Run order (by start time): g=5: ")


def test_superseded_note_lists_the_kept_records(tmp_path):
    (tmp_path / "runs" / "g5.0").mkdir(parents=True)
    old = tmp_path / "runs" / "round1-cli-omp3"
    (old / "g5.0" / "cli-warm1").mkdir(parents=True)
    (old / "g5.0" / "cli-warm1" / "record.json").write_text("{}")
    (old / "README.txt").write_text("Round 1 CLI launches, superseded by D82.\nmore\n")
    assert cw.superseded_note(tmp_path) == [
        "Superseded records kept under runs/round1-cli-omp3/ (not used here): g5.0/cli-warm1 — "
        "Round 1 CLI launches, superseded by D82."]


def test_demote_headings_and_drop_section():
    text = ("# A\n## B\nx\n```bash\n# not a heading\n```\n## Reproduction\ny\n```bash\n"
            "# a comment in the dropped block\n```\n### Sub\nz\n## C\n")
    assert nr.demote_headings("# A\n```\n# c\n```\n## B") == "## A\n```\n# c\n```\n### B"
    assert nr.drop_section(text, "## Reproduction") == (
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
    cli = cw.Launch("cli-warm1", "cli", 5.0, "cli-default", "warm")
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
    lines = nr.check_lines(checks)
    assert lines[0] == ("- `script-warm1`: 2/3 pass; counter_rule_warm: r0 0/39/7459; "
                        "**flag** step_series_it2: p90/median 1.2")
    assert lines[1] == ("- `cli-warm1`: 0/1 pass; file_rule_warm: new .neff 1; **FAIL** "
                        "file_rule_warm: new .neff 1")


def test_table_md_escapes_pipes():
    md = nr.table_md([["a|b", "c", "d", "e", "f", "g"]])
    assert "a\\|b" in md and md.count("\n") == 2
