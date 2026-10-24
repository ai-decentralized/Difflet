"""The benchmark result cache (``benchmark/result_cache.py``, stored in
``benchmark/cached_results/<slug>.json``) and its use by ``bench.py`` and ``neuron_cold_warm.py``:
CPU only, no device, no network (the host probes are stubbed), the store under ``tmp_path``."""

from __future__ import annotations

import argparse
import datetime
import json
import sys

import pytest

from benchmark import bench, report, result_cache as rc
from benchmark import neuron_cold_warm as cw
from benchmark.models import MATRIX
from tests.unit.benchmark.test_neuron_cold_warm import _records, _write_records

T0 = datetime.datetime(2026, 10, 1, 12, 0, tzinfo=datetime.timezone.utc)
CODE = {"git_head": "a" * 40, "difflet_tree": "t" * 40, "dirty": False}
PLATFORM = {"instance_type": "trn2.3xlarge", "device": "trn2.3xlarge / 4 NeuronCores",
            "sdk": {"neuronx-cc": "2.21.0", "torch-neuronx": "2.9.0",
                    "aws-neuronx-runtime-lib": "2.28.0"}}
SLUG = "flux_1_dev"


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """Every test gets its own store; the committed benchmark/cached_results/ is never read."""
    path = tmp_path / "cached_results"
    monkeypatch.setattr(rc, "STORE_DIR", path)
    return path


def _fp(platform=None, **run_options):
    config = rc.config_identity(MATRIX[SLUG], backend="trainium",
                                run_options={"iters": 0, "natural_iters": 2, **run_options})
    return rc.fingerprint(config, platform or PLATFORM)


def _result(status="ok", **extra):
    return {"status": status, "model_id": "m", "backend": "trainium", "compile_seconds": 1484.0,
            **extra}


def _entry(fp=None, *, status="ok", measured_at=T0, **extra):
    return {"measured_at": rc.iso(measured_at), "fingerprint": fp or _fp(), "code": dict(CODE),
            "result": _result(status, **extra)}


def _days(n: float) -> datetime.datetime:
    return T0 + datetime.timedelta(days=n)


def _save(fp=None, *, measured_at=T0, now=None, **extra):
    return rc.save(SLUG, fp or _fp(), _result(**extra), code=CODE, measured_at=measured_at,
                   now=now or measured_at)


# ------------------------------------------------------------------------------ the store


def test_one_json_per_model_with_one_entry_per_platform(store):
    _save()
    trn3 = {**PLATFORM, "instance_type": "trn3pd98.3xlarge", "device": "trn3"}
    _save(_fp(trn3), measured_at=_days(1))
    assert sorted(p.name for p in store.iterdir()) == ["flux_1_dev.json"]
    doc = json.loads((store / "flux_1_dev.json").read_text())
    assert doc["schema"] == 1 and doc["slug"] == SLUG
    assert set(doc["entries"]) == {rc.key_of(_fp()), rc.key_of(_fp(trn3))}
    assert doc["entries"][rc.key_of(_fp())]["result"]["compile_seconds"] == 1484.0
    # each platform finds its own entry
    assert rc.decide(SLUG, _fp(), now=_days(2)).reuse
    assert rc.decide(SLUG, _fp(trn3), now=_days(2)).reuse


def test_re_measuring_replaces_the_entry_and_drops_expired_ones(store):
    other = {**PLATFORM, "instance_type": "trn2.48xlarge"}
    _save(_fp(other))                                    # will be expired at day 40
    _save(measured_at=_days(20), compile_seconds=1.0)
    _save(measured_at=_days(40), compile_seconds=2.0)    # same key: replaced
    ents = rc.entries(SLUG)
    assert list(ents) == [rc.key_of(_fp())]
    assert ents[rc.key_of(_fp())]["result"]["compile_seconds"] == 2.0


def test_only_completed_results_are_stored(store):
    for status in ("failed", "partial"):
        with pytest.raises(ValueError):
            _save(status=status)
        assert rc.record(SLUG, _fp(), _result(status), code=CODE, measured_at=T0) is None
    by_g = _result(results_by_guidance={"5.0": {"status": "partial"}})
    assert rc.record(SLUG, _fp(), by_g, code=CODE, measured_at=T0) is None
    assert not store.exists()


def test_record_stores_and_adds_the_banner_summary(store, monkeypatch):
    monkeypatch.setattr(rc, "utc_now", lambda: T0)
    doc = _result()
    entry = rc.record(SLUG, _fp(), doc, code=CODE, measured_at=T0)
    assert "cache" not in entry["result"]  # the store holds the measurement, not the summary
    assert doc["cache"] == {"key": rc.key_of(_fp()), "measured_at": "2026-10-01T12:00:00Z",
                            "instance_type": "trn2.3xlarge", "git_head": "a" * 40}


# ------------------------------------------------------------------------------ decide


def test_a_fresh_matching_result_is_reused():
    _save()
    d = rc.decide(SLUG, _fp(), now=_days(3))
    assert d.reuse and d.key == rc.key_of(_fp())
    assert "trn2.3xlarge" in d.reason and "expires in 27.0 days" in d.reason


def test_no_stored_result():
    d = rc.decide(SLUG, _fp(), now=_days(1))
    assert not d.reuse and d.reason == "no stored result in flux_1_dev.json"


@pytest.mark.parametrize("entry, why", [
    (_entry(status="failed"), "status is 'failed'"),
    (_entry(status="partial"), "status is 'partial'"),
    (_entry(results_by_guidance={"5.0": {"status": "partial"}}), "status is 'partial'"),
])
def test_an_unsuccessful_entry_is_never_reused(entry, why):
    d = rc.decide_entry(entry, _fp(), now=_days(1))
    assert not d.reuse and why in d.reason


def test_a_flagged_run_is_reusable():
    entry = _entry(status="flagged", results_by_guidance={"5.0": {"status": "flagged"},
                                                          "1.0": {"status": "ok"}})
    assert rc.decide_entry(entry, _fp(), now=_days(1)).reuse


def test_a_different_instance_type_forces_a_re_run():
    _save()
    other = {**PLATFORM, "instance_type": "trn2.48xlarge"}
    d = rc.decide(SLUG, _fp(other), now=_days(1))
    assert not d.reuse
    assert d.reason.startswith("instance type changed")
    assert "'trn2.3xlarge' -> 'trn2.48xlarge'" in d.reason


@pytest.mark.parametrize("sdk_change", [
    {"neuronx-cc": "2.22.0"},                          # a compiler upgrade
    {"aws-neuronx-runtime-lib": "2.29.0"},             # the OS-level runtime
    {"nki": "0.3.0"},                                  # a package that was not there
])
def test_a_different_sdk_forces_a_re_run(sdk_change):
    _save()
    other = {**PLATFORM, "sdk": {**PLATFORM["sdk"], **sdk_change}}
    d = rc.decide(SLUG, _fp(other), now=_days(1))
    assert not d.reuse and d.reason.startswith("SDK changed: sdk.")


def test_unreadable_sdk_is_never_a_hit():
    blank = {**PLATFORM, "sdk": {}}
    d = rc.decide_entry(_entry(_fp(blank)), _fp(blank), now=_days(1))
    assert not d.reuse and "SDK versions could not be read" in d.reason


@pytest.mark.parametrize("instance", ["unknown", "", None])
def test_an_unknown_instance_type_never_matches(instance):
    blind = {**PLATFORM, "instance_type": instance}
    d = rc.decide_entry(_entry(_fp(blind)), _fp(blind), now=_days(1))
    assert not d.reuse and "DIFFLET_INSTANCE_TYPE" in d.reason


def test_instance_type_override(monkeypatch):
    monkeypatch.setenv("DIFFLET_INSTANCE_TYPE", "lambda-1xB300")
    assert rc.instance_type() == "lambda-1xB300"


def test_a_different_config_or_run_option_forces_a_re_run():
    _save()
    d = rc.decide(SLUG, _fp(iters=3), now=_days(1))
    assert not d.reuse and "run_options.iters: 0 -> 3" in d.reason


def test_a_job_missing_a_requested_guidance_is_not_reused():
    fp = _fp(guidance=[5.0, 1.0])
    entry = _entry(fp, results_by_guidance={"5.0": {"status": "ok"}})
    d = rc.decide_entry(entry, fp, now=_days(1))
    assert not d.reuse and d.reason == "stored result is missing guidance 1.0"
    entry["result"]["results_by_guidance"]["1.0"] = {"status": "ok"}
    assert rc.decide_entry(entry, fp, now=_days(1)).reuse


def test_results_expire_after_thirty_days_and_reuse_does_not_refresh_them():
    _save()
    assert rc.decide(SLUG, _fp(), now=_days(29.9)).reuse
    d = rc.decide(SLUG, _fp(), now=_days(30.1))
    assert not d.reuse and d.reason.startswith("expired: measured 30.1 days ago")
    assert not rc.decide(SLUG, _fp(), now=_days(8), max_age_days=7).reuse
    entry = rc.mark_reused(SLUG, rc.key_of(_fp()), now=_days(20), platform=PLATFORM, code=CODE)
    assert entry["measured_at"] == rc.iso(T0)  # the clock is the measurement's
    assert not rc.decide(SLUG, _fp(), now=_days(31)).reuse


def test_code_changes_only_matter_when_asked():
    _save()
    moved = {**CODE, "difflet_tree": "u" * 40}
    assert rc.decide(SLUG, _fp(), now=_days(1), code=moved).reuse
    d = rc.decide(SLUG, _fp(), now=_days(1), code=moved, rerun_on_code_change=True)
    assert not d.reuse and "difflet/ code changed" in d.reason
    dirty = {**CODE, "dirty": True}
    assert not rc.decide(SLUG, _fp(), now=_days(1), code=dirty, rerun_on_code_change=True).reuse


# ------------------------------------------------------------------------------ identity


def test_config_identity_drops_labels_and_keeps_what_runs():
    cfg = rc.config_identity(MATRIX[SLUG], backend="trainium", run_options={"iters": 2})
    assert cfg["backend"] == "trainium" and cfg["run_options"] == {"iters": 2}
    assert cfg["revision"] == MATRIX[SLUG].revision and cfg["steps"] == 28
    for label in rc.PRESENTATION_FIELDS:
        assert label not in cfg


def test_the_key_does_not_depend_on_dict_order():
    a = rc.fingerprint({"x": 1, "y": {"b": 2, "a": 1}}, PLATFORM)
    b = rc.fingerprint({"y": {"a": 1, "b": 2}, "x": 1},
                       dict(reversed(list(PLATFORM.items()))))
    assert rc.key_of(a) == rc.key_of(b) and len(rc.key_of(a)) == 16
    assert rc.key_of(a) != rc.key_of(rc.fingerprint({"x": 2}, PLATFORM))


def test_platform_identity_merges_python_and_system_sdk():
    class Adapter:
        def toolchain(self):
            return {"torch": "2.9"}

        def device_info(self):
            return "dev"

    p = rc.platform_identity(Adapter(), instance="trn2.3xlarge",
                             system={"aws-neuronx-dkms": "2.24"})
    assert p == {"instance_type": "trn2.3xlarge", "device": "dev",
                 "sdk": {"aws-neuronx-dkms": "2.24", "torch": "2.9"}}


def test_iso_round_trips():
    assert rc.parse_iso(rc.iso(T0)) == T0


# ------------------------------------------------------------------------------ the reports


def test_try_reuse_rewrites_the_report_from_the_store_only_on_a_hit(tmp_path):
    _save()
    path, md = tmp_path / "trn2" / "r.json", tmp_path / "trn2" / "r.md"
    miss = rc.try_reuse(SLUG, _fp(iters=1), now=_days(1), max_age_days=30, code=CODE,
                        rerun_on_code_change=False, render=lambda d: "x", json_path=path,
                        md_path=md)
    assert not miss.reuse and not path.exists()
    assert "last_reused" not in rc.entries(SLUG)[rc.key_of(_fp())]
    later = {**CODE, "git_head": "b" * 40}
    hit = rc.try_reuse(SLUG, _fp(), now=_days(2), max_age_days=30, code=later,
                       rerun_on_code_change=False, render=lambda d: "\n".join(rc.banner(d)),
                       json_path=path, md_path=md)
    assert hit.reuse
    reused = {"at": "2026-10-03T12:00:00Z", "instance_type": "trn2.3xlarge",
              "git_head": "b" * 40}
    assert rc.entries(SLUG)[rc.key_of(_fp())]["last_reused"] == reused
    doc = json.loads(path.read_text())
    assert doc["compile_seconds"] == 1484.0 and doc["cache"]["last_reused"] == reused
    assert "**REUSED:**" in md.read_text() and "bbbbbbbbbbbb" in md.read_text()


def test_report_says_when_it_was_measured_and_that_it_was_reused():
    entry = _entry()
    doc = rc.report_doc(entry)
    text = report.render(doc)
    assert "**Measured:** 2026-10-01T12:00:00Z on trn2.3xlarge at `aaaaaaaaaaaa`" in text
    assert "cached in `benchmark/cached_results/`" in text
    assert "reusable until 2026-10-31T12:00:00Z at the default 30-day limit" in text
    assert "REUSED" not in text
    entry["last_reused"] = {"at": rc.iso(_days(5)), "instance_type": "trn2.3xlarge",
                            "git_head": "c" * 40}
    assert "**REUSED:** these numbers were NOT re-measured" in report.render(rc.report_doc(entry))
    assert "Measured" not in report.render({"model_id": "m", "backend": "x"})  # legacy report


def test_status_rows(store):
    _save()
    _save(_fp({**PLATFORM, "instance_type": "trn2.48xlarge"}), measured_at=_days(-40),
          now=_days(-40))
    rows = rc.status_rows(now=_days(2))
    assert [slug for slug, _ in rows] == [SLUG, SLUG]
    states = sorted(state for _, state in rows)
    assert states[0].startswith("EXPIRED (42.0 days old; trn2.48xlarge, trainium, key ")
    assert states[1].startswith("fresh (2.0 days old, expires in 28.0 days; trn2.3xlarge")


# ------------------------------------------------------------------------------ patch scripts


def test_patch_scripts_drop_the_report_summary_but_not_the_store(tmp_path, monkeypatch):
    from benchmark import cold_warm_e2e, models

    _save()
    monkeypatch.setattr(models, "_RESULTS_ROOT", str(tmp_path))
    monkeypatch.setattr(models, "DEVICE", "trn2")
    (tmp_path / "trn2").mkdir()
    (tmp_path / "trn2" / "flux_1_dev.json").write_text(json.dumps(rc.report_doc(_entry())))
    cold_warm_e2e._patch(SLUG, e2e_cold_seconds=1.0, status="ok")
    patched = json.loads((tmp_path / "trn2" / "flux_1_dev.json").read_text())
    assert "cache" not in patched and patched["e2e_cold_seconds"] == 1.0
    assert "**Measured:**" not in (tmp_path / "trn2" / "flux_1_dev.md").read_text()
    # the store keeps the measurement as it was measured
    assert "e2e_cold_seconds" not in rc.entries(SLUG)[rc.key_of(_fp())]["result"]
    for name in ("step_latency", "warm_e2e", "step_realloop", "backfill_breakdowns"):
        source = (rc.REPO_ROOT / "benchmark" / f"{name}.py").read_text()
        assert "result_cache.invalidate(d)" in source, name


# ------------------------------------------------------------------------------ bench.py


class FakeAdapter:
    supports_natural_mode = False
    toolchain_versions = {"neuronx-cc": "2.21.0", "torch-neuronx": "2.9.0"}
    generates = 0

    def device_info(self):
        return "trn2.3xlarge / 4 NeuronCores"

    def toolchain(self):
        return dict(FakeAdapter.toolchain_versions)

    def prepare(self, spec):
        pass

    def compile(self, spec):
        return 1484.0, {"transformer": 1484.0}

    def run_generate(self, spec, **kwargs):
        FakeAdapter.generates += 1
        return {"wall_seconds": 40.0, "load_seconds": 30.0, "step_seconds": [0.3, 0.31]}


@pytest.fixture
def bench_env(tmp_path, monkeypatch, store):
    from benchmark import models

    FakeAdapter.generates = 0
    FakeAdapter.toolchain_versions = {"neuronx-cc": "2.21.0", "torch-neuronx": "2.9.0"}
    clock = {"now": T0}
    monkeypatch.setattr(models, "_RESULTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(models, "DEVICE", "trn2")
    monkeypatch.setattr(bench, "_make_adapter", lambda name: FakeAdapter())
    monkeypatch.setattr(rc, "instance_type", lambda: "trn2.3xlarge")
    monkeypatch.setattr(rc, "system_sdk", lambda: {"aws-neuronx-runtime-lib": "2.28.0"})
    monkeypatch.setattr(rc, "code_identity", lambda repo=None: dict(CODE))
    monkeypatch.setattr(rc, "utc_now", lambda: clock["now"])

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["bench", "--model", SLUG, *extra])
        before = FakeAdapter.generates
        assert bench.main() == 0
        return FakeAdapter.generates - before

    run.clock = clock
    run.json = tmp_path / "reports" / "trn2" / "flux_1_dev.json"
    run.md = tmp_path / "reports" / "trn2" / "flux_1_dev.md"
    run.store = store / "flux_1_dev.json"
    return run


def test_bench_measures_once_then_reuses_on_the_next_instance(bench_env, capsys):
    assert bench_env() == 1  # measured: the cold generate
    [entry] = json.loads(bench_env.store.read_text())["entries"].values()
    assert entry["result"]["compile_seconds"] == 1484.0
    assert entry["measured_at"] == "2026-10-01T12:00:00Z"
    assert entry["fingerprint"]["platform"]["sdk"]["aws-neuronx-runtime-lib"] == "2.28.0"
    assert json.loads(bench_env.json.read_text())["cache"]["measured_at"] == entry["measured_at"]
    assert "REUSED" not in bench_env.md.read_text()

    # a new instance: the reports are gone, only the committed store is there
    bench_env.json.unlink()
    bench_env.md.unlink()
    bench_env.clock["now"] = _days(5)
    assert bench_env() == 0  # nothing ran
    out = capsys.readouterr().out
    assert "[cache] flux_1_dev: REUSE -- measured 2026-10-01T12:00:00Z on trn2.3xlarge" in out
    doc = json.loads(bench_env.json.read_text())
    assert doc["compile_seconds"] == 1484.0
    assert doc["cache"]["last_reused"]["at"] == rc.iso(_days(5))
    assert "**REUSED:**" in bench_env.md.read_text()


def test_bench_re_measures_on_sdk_change_expiry_force_and_options(bench_env, capsys):
    assert bench_env() == 1
    assert bench_env("--iters", "2") == 3  # a different run option: cold + 2 warm
    assert bench_env("--iters", "2") == 0
    assert bench_env("--iters", "2", "--force") == 3
    FakeAdapter.toolchain_versions = {**FakeAdapter.toolchain_versions, "neuronx-cc": "2.22.0"}
    assert bench_env("--iters", "2") == 3
    assert "RUN -- SDK changed: sdk.neuronx-cc: '2.21.0' -> '2.22.0'" in capsys.readouterr().out
    bench_env.clock["now"] = _days(31)
    assert bench_env("--iters", "2") == 3
    assert "RUN -- expired" in capsys.readouterr().out
    ents = json.loads(bench_env.store.read_text())["entries"]
    assert [e["measured_at"] for e in ents.values()] == [rc.iso(_days(31))]  # expired dropped
    assert "last_reused" not in next(iter(ents.values()))


def test_bench_plan_runs_and_writes_nothing(bench_env, capsys):
    assert bench_env("--plan") == 0
    assert not bench_env.json.exists() and not bench_env.store.exists()
    assert "[cache] flux_1_dev: RUN -- no stored result" in capsys.readouterr().out
    bench_env()
    before = bench_env.store.read_bytes(), bench_env.json.read_bytes()
    assert bench_env("--plan") == 0
    assert (bench_env.store.read_bytes(), bench_env.json.read_bytes()) == before
    assert "REUSE -- measured" in capsys.readouterr().out


def test_a_failed_run_is_not_stored_and_is_retried(bench_env, monkeypatch, capsys):
    compiles = []

    def broken(self, spec):
        compiles.append(spec)
        raise RuntimeError("compile failed")

    monkeypatch.setattr(FakeAdapter, "compile", broken)
    bench_env()
    doc = json.loads(bench_env.json.read_text())
    assert doc["status"] == "failed" and "cache" not in doc
    assert not bench_env.store.exists()
    bench_env()
    assert len(compiles) == 2  # attempted again


def test_bench_skip_compile_is_part_of_the_key(bench_env, capsys):
    bench_env("--skip-compile")
    assert json.loads(bench_env.json.read_text())["compile_seconds"] is None
    assert bench_env() == 1  # a full run does not take the compile-less result
    assert "run_options.skip_compile: True -> False" in capsys.readouterr().out


def test_bench_leaves_the_neuron_jobs_report_alone(bench_env, capsys):
    bench_env.json.parent.mkdir(parents=True)
    owned = {"status": "ok", "results_by_guidance": {"1.0": {"status": "ok"}}}
    bench_env.json.write_text(json.dumps(owned))
    assert bench_env() == 0
    assert "SKIP -- " in capsys.readouterr().out
    assert json.loads(bench_env.json.read_text()) == owned
    assert bench_env("--force") == 1  # explicit overwrite


# ------------------------------------------------------------------------------ neuron job


def _job_fp(true_cold=False, platform=None):
    args = argparse.Namespace(guidance=list(cw.GUIDANCES), iters=3, warm_launches=3,
                              cli_warm=3, no_w4_check=False, steps=None)
    return rc.fingerprint(rc.config_identity(MATRIX[cw.SLUG], backend="neuron",
                                             run_options=cw.run_options(args,
                                                                        true_cold=true_cold)),
                          platform or PLATFORM)


def _assembled(tmp_path, *, fp):
    records = {**_records(5.0), **_records(1.0)}
    _write_records(tmp_path / "rec", records)
    out_json, out_md = tmp_path / "wan.json", tmp_path / "wan.md"
    doc = cw.assemble(cw.load_records(tmp_path / "rec"), adapter=None, out_json=out_json,
                      out_md=out_md, fingerprint=fp)
    return doc, out_json, out_md


@pytest.fixture
def job_env(monkeypatch):
    monkeypatch.setattr(rc, "code_identity", lambda repo=None: dict(CODE))
    monkeypatch.setattr(rc, "platform_identity", lambda adapter, **kw: dict(PLATFORM))
    monkeypatch.setattr(cw, "sudo_available", lambda: False)
    handler = cw.signal.getsignal(cw.signal.SIGTERM)
    yield
    cw.signal.signal(cw.signal.SIGTERM, handler)


def test_a_completed_job_is_stored_as_old_as_its_oldest_launch(tmp_path, store, job_env):
    doc, out_json, out_md = _assembled(tmp_path, fp=_job_fp())
    records = cw.load_records(tmp_path / "rec")
    first = min(r["run"].get("start_epoch") or r["run"]["end_epoch"] for r in records.values())
    [entry] = rc.entries(cw.SLUG).values()
    assert entry["measured_at"] == rc.iso(datetime.datetime.fromtimestamp(
        first, datetime.timezone.utc))
    assert entry["result"]["results_by_guidance"] == doc["results_by_guidance"]
    assert json.loads(out_json.read_text())["cache"]["key"] == rc.key_of(_job_fp())
    text = out_md.read_text()
    assert text.startswith("# Benchmark — Wan2.1")
    assert "\n\n**Measured:** " in text and "**Status:**" in text
    # no fingerprint (--assemble-only): nothing stored, the old report
    plain = cw.assemble(records, adapter=None, out_json=tmp_path / "p.json",
                        out_md=tmp_path / "p.md")
    assert "cache" not in plain and "**Measured:**" not in (tmp_path / "p.md").read_text()


def test_neuron_job_reuses_the_stored_result_without_launching(tmp_path, monkeypatch, capsys,
                                                               job_env):
    def no_launch(*a, **kw):
        raise AssertionError("a launch ran although the result was reusable")

    monkeypatch.setattr(cw, "run_launch", no_launch)
    monkeypatch.setattr(cw, "host_info", no_launch)
    doc, _, _ = _assembled(tmp_path, fp=_job_fp())
    measured = rc.parse_iso(next(iter(rc.entries(cw.SLUG).values()))["measured_at"])
    monkeypatch.setattr(rc, "utc_now", lambda: measured + datetime.timedelta(days=2))
    new_json, new_md = tmp_path / "new" / "wan.json", tmp_path / "new" / "wan.md"
    argv = ["--record-dir", str(tmp_path / "rec2"), "--true-cold", "no",
            "--out-json", str(new_json), "--out-md", str(new_md)]

    assert cw.main(argv) == 0
    assert "[cache] REUSE -- measured" in capsys.readouterr().out
    reused = json.loads(new_json.read_text())
    assert reused["cache"]["last_reused"]["instance_type"] == "trn2.3xlarge"
    assert reused["results_by_guidance"] == doc["results_by_guidance"]
    assert "**REUSED:**" in new_md.read_text() and "## Headline" in new_md.read_text()

    # --true-cold yes is another measurement: the job runs (here: its first launch raises)
    with pytest.raises(AssertionError, match="a launch ran"):
        cw.main([*argv[:3], "yes", *argv[4:]])
    # and so is --force
    with pytest.raises(AssertionError, match="a launch ran"):
        cw.main([*argv, "--force"])


def test_a_stopped_neuron_job_is_not_stored(tmp_path, monkeypatch, job_env):
    monkeypatch.setattr(cw, "host_info", lambda adapter: {"device": "d", "toolchain": {}})

    def interrupted(launch, **kwargs):
        raise KeyboardInterrupt("signal 2")

    monkeypatch.setattr(cw, "run_launch", interrupted)
    out_json = tmp_path / "wan.json"
    code = cw.main(["--record-dir", str(tmp_path / "rec"), "--true-cold", "no", "--force",
                    "--out-json", str(out_json), "--out-md", str(tmp_path / "wan.md")])
    assert code == 130
    assert rc.entries(cw.SLUG) == {}


def test_a_resume_finishes_the_job_instead_of_reusing(tmp_path, monkeypatch, capsys, job_env):
    monkeypatch.setattr(cw, "host_info", lambda adapter: {"device": "d", "toolchain": {}})

    def consulted(*a, **kw):
        raise AssertionError("the stored result was consulted on a resume")

    monkeypatch.setattr(rc, "try_reuse", consulted)

    def interrupted(launch, **kwargs):
        raise KeyboardInterrupt("signal 2")

    monkeypatch.setattr(cw, "run_launch", interrupted)
    _write_records(tmp_path / "rec", {"g5.0/script-cold": _records(5.0)["g5.0/script-cold"]})
    # the bare record has no rank files, so re-judging it fails after the cache decision
    with pytest.raises(ValueError, match="expected result-rank"):
        cw.main(["--record-dir", str(tmp_path / "rec"), "--true-cold", "no", "--skip-existing",
                 "--out-json", str(tmp_path / "o.json"), "--out-md", str(tmp_path / "o.md")])
    assert "[cache] RUN -- resuming the launches recorded in" in capsys.readouterr().out


def test_skip_existing_keeps_only_records_from_this_host_and_age():
    now = datetime.datetime.fromtimestamp(10_000.0, datetime.timezone.utc)
    host = {"device": "trn2.3xlarge", "toolchain": {"torch": "2.14"}}
    rec = {"ok": True, "host": dict(host), "run": {"start_epoch": 9_000.0}}
    assert cw.reusable_record(rec, host, now=now, max_age_days=30) is None
    assert cw.reusable_record({**rec, "ok": False}, host, now=now, max_age_days=30) == "not ok"
    other = {**rec, "host": {**host, "toolchain": {"torch": "2.15"}}}
    assert cw.reusable_record(other, host, now=now, max_age_days=30) == (
        "recorded on another toolchain")
    later = now + datetime.timedelta(days=31)
    assert cw.reusable_record(rec, host, now=later, max_age_days=30) == "older than 30 days"
    assert cw.reusable_record({"ok": True}, host, now=now, max_age_days=30) is None  # pre-M-2


def test_assemble_only_keeps_the_summary_of_the_same_records(tmp_path, job_env):
    doc, out_json, out_md = _assembled(tmp_path, fp=_job_fp())
    records = cw.load_records(tmp_path / "rec")
    again = cw.assemble(records, adapter=None, out_json=out_json, out_md=out_md,
                        cache=doc["cache"])
    assert again["cache"] == doc["cache"] and "**Measured:**" in out_md.read_text()
    stale = {**doc["cache"], "measured_at": "2020-01-01T00:00:00Z"}  # other records' summary
    assert "cache" not in cw.assemble(records, adapter=None, out_json=out_json, out_md=out_md,
                                      cache=stale)


# ------------------------------------------------------------------------------ second review


def test_a_stricter_max_age_never_deletes_another_platforms_entry(store, monkeypatch):
    trn3 = {**PLATFORM, "instance_type": "trn3pd98.3xlarge", "device": "trn3"}
    _save(_fp(trn3))                                     # 20 days old at the trn2 run below
    monkeypatch.setattr(rc, "utc_now", lambda: _days(20))
    rc.record(SLUG, _fp(), _result(), code=CODE, measured_at=_days(20))
    assert rc.key_of(_fp(trn3)) in rc.entries(SLUG)
    assert rc.decide(SLUG, _fp(trn3), now=_days(21)).reuse


def test_a_miss_is_explained_against_the_same_platform():
    _save()                                               # trn2, iters 0
    trn3 = {**PLATFORM, "instance_type": "trn3pd98.3xlarge", "device": "trn3"}
    _save(_fp(trn3), measured_at=_days(1))                # newer, another platform
    d = rc.decide(SLUG, _fp(iters=3), now=_days(2))
    assert not d.reuse and d.reason.startswith("config changed: run_options.iters: 0 -> 3")


def test_compiler_flags_from_the_shell_are_part_of_the_config():
    plain = rc.config_identity(MATRIX[SLUG], backend="trainium", run_options={}, env={})
    flagged = rc.config_identity(MATRIX[SLUG], backend="trainium", run_options={},
                                 env={"NEURON_CC_FLAGS": "--optlevel 1", "HOME": "/x"})
    assert plain["env"] == {} and flagged["env"] == {"NEURON_CC_FLAGS": "--optlevel 1"}
    d = rc.decide_entry(_entry(rc.fingerprint(plain, PLATFORM)),
                        rc.fingerprint(flagged, PLATFORM), now=_days(1))
    assert not d.reuse and "env.NEURON_CC_FLAGS" in d.reason


def test_a_failed_re_measurement_drops_the_older_entry(store, monkeypatch):
    monkeypatch.setattr(rc, "utc_now", lambda: _days(1))
    _save()
    other = {**PLATFORM, "instance_type": "trn2.48xlarge"}
    _save(_fp(other))
    assert rc.record(SLUG, _fp(), _result("failed"), code=CODE, measured_at=_days(1)) is None
    assert list(rc.entries(SLUG)) == [rc.key_of(_fp(other))]  # only that key
    assert not rc.decide(SLUG, _fp(), now=_days(2)).reuse


def test_skip_existing_does_not_keep_a_launch_from_another_fingerprint():
    now = datetime.datetime.fromtimestamp(10_000.0, datetime.timezone.utc)
    host = {"device": "d", "toolchain": {}}
    rec = {"ok": True, "host": dict(host), "run": {"start_epoch": 9_000.0}, "cache_key": "aaa"}
    assert cw.reusable_record(rec, host, now=now, max_age_days=30, cache_key="aaa") is None
    assert cw.reusable_record(rec, host, now=now, max_age_days=30, cache_key="bbb") == (
        "recorded with other options or on another platform")
