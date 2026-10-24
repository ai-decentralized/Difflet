"""The committed-result cache (``benchmark/result_cache.py``) and its use by ``bench.py`` and
``neuron_cold_warm.py``: CPU only, no device, no network (the host probes are stubbed)."""

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


def _fp(platform=None, **run_options):
    config = rc.config_identity(MATRIX["flux_1_dev"], backend="trainium",
                                run_options={"iters": 0, "natural_iters": 2, **run_options})
    return rc.fingerprint(config, platform or PLATFORM)


def _doc(fp=None, *, status="ok", measured_at=T0, code=CODE):
    return {"status": status, "model_id": "m",
            "cache": rc.cache_block(fp or _fp(), code=code, measured_at=measured_at)}


def _days(n: float) -> datetime.datetime:
    return T0 + datetime.timedelta(days=n)


# ------------------------------------------------------------------------------ decide


def test_a_fresh_matching_result_is_reused():
    d = rc.decide(_doc(), _fp(), now=_days(3))
    assert d.reuse and "trn2.3xlarge" in d.reason and "expires in 27.0 days" in d.reason


@pytest.mark.parametrize("doc, why", [
    (None, "no committed result"),
    ({"status": "ok", "model_id": "m"}, "no cache fingerprint"),
    (_doc(status="failed"), "status is 'failed'"),
    (_doc(status="partial"), "status is 'partial'"),
])
def test_missing_legacy_or_unsuccessful_results_are_re_measured(doc, why):
    d = rc.decide(doc, _fp(), now=_days(1))
    assert not d.reuse and why in d.reason


def test_a_flagged_run_is_reusable_but_a_partial_guidance_is_not():
    doc = _doc(status="flagged")
    doc["results_by_guidance"] = {"5.0": {"status": "flagged"}, "1.0": {"status": "ok"}}
    assert rc.decide(doc, _fp(), now=_days(1)).reuse
    doc["results_by_guidance"]["5.0"]["status"] = "partial"
    d = rc.decide(doc, _fp(), now=_days(1))
    assert not d.reuse and "'partial'" in d.reason


def test_a_different_instance_type_forces_a_re_run():
    other = {**PLATFORM, "instance_type": "trn2.48xlarge"}
    d = rc.decide(_doc(), _fp(other), now=_days(1))
    assert not d.reuse
    assert d.reason.startswith("instance type changed")
    assert "'trn2.3xlarge' -> 'trn2.48xlarge'" in d.reason


@pytest.mark.parametrize("sdk_change", [
    {"neuronx-cc": "2.22.0"},                          # a compiler upgrade
    {"aws-neuronx-runtime-lib": "2.29.0"},             # the OS-level runtime
    {"nki": "0.3.0"},                                  # a package that was not there
])
def test_a_different_sdk_forces_a_re_run(sdk_change):
    other = {**PLATFORM, "sdk": {**PLATFORM["sdk"], **sdk_change}}
    d = rc.decide(_doc(), _fp(other), now=_days(1))
    assert not d.reuse and d.reason.startswith("SDK changed: sdk.")


def test_unreadable_sdk_is_never_a_hit():
    blank = {**PLATFORM, "sdk": {}}
    d = rc.decide(_doc(_fp(blank)), _fp(blank), now=_days(1))
    assert not d.reuse and "SDK versions could not be read" in d.reason


def test_a_different_config_or_run_option_forces_a_re_run():
    d = rc.decide(_doc(), _fp(iters=3), now=_days(1))
    assert not d.reuse and "run_options.iters: 0 -> 3" in d.reason


def test_results_expire_after_thirty_days_and_reuse_does_not_refresh_them():
    assert rc.decide(_doc(), _fp(), now=_days(29.9)).reuse
    d = rc.decide(_doc(), _fp(), now=_days(30.1))
    assert not d.reuse and d.reason.startswith("expired: measured 30.1 days ago")
    assert not rc.decide(_doc(), _fp(), now=_days(8), max_age_days=7).reuse
    doc = rc.mark_reused(_doc(), now=_days(20), platform=PLATFORM, code=CODE)
    assert doc["cache"]["measured_at"] == rc.iso(T0)  # the clock is the measurement's
    assert not rc.decide(doc, _fp(), now=_days(31)).reuse


def test_code_changes_only_matter_when_asked():
    moved = {**CODE, "difflet_tree": "u" * 40}
    assert rc.decide(_doc(), _fp(), now=_days(1), code=moved).reuse
    d = rc.decide(_doc(), _fp(), now=_days(1), code=moved, rerun_on_code_change=True)
    assert not d.reuse and "difflet/ code changed" in d.reason
    dirty = {**CODE, "dirty": True}
    assert not rc.decide(_doc(), _fp(), now=_days(1), code=dirty,
                         rerun_on_code_change=True).reuse


# ------------------------------------------------------------------------------ identity


def test_config_identity_drops_labels_and_keeps_what_runs():
    cfg = rc.config_identity(MATRIX["flux_1_dev"], backend="trainium",
                             run_options={"iters": 2})
    assert cfg["backend"] == "trainium" and cfg["run_options"] == {"iters": 2}
    assert cfg["revision"] == MATRIX["flux_1_dev"].revision and cfg["steps"] == 28
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
    block = rc.cache_block(_fp(), code=CODE, measured_at=T0)
    assert block["measured_at"] == "2026-10-01T12:00:00Z"
    assert block["expires_at"] == "2026-10-31T12:00:00Z"
    assert block["key"] == rc.key_of(_fp())


# ------------------------------------------------------------------------------ reuse marking


def test_try_reuse_marks_and_re_renders_only_on_a_hit(tmp_path):
    path, md = tmp_path / "r.json", tmp_path / "r.md"
    path.write_text(json.dumps(_doc()))
    miss = rc.try_reuse(path, _fp(iters=1), now=_days(1), max_age_days=30, code=CODE,
                        rerun_on_code_change=False, render=lambda d: "x", md_path=md)
    assert not miss.reuse and not md.exists()
    assert "last_reused" not in json.loads(path.read_text())["cache"]
    later = {**CODE, "git_head": "b" * 40}
    hit = rc.try_reuse(path, _fp(), now=_days(2), max_age_days=30, code=later,
                       rerun_on_code_change=False,
                       render=lambda d: "\n".join(rc.banner(d)), md_path=md)
    assert hit.reuse
    doc = json.loads(path.read_text())
    assert doc["cache"]["last_reused"] == {"at": "2026-10-03T12:00:00Z",
                                           "instance_type": "trn2.3xlarge",
                                           "git_head": "b" * 40}
    assert "**REUSED:**" in md.read_text() and "bbbbbbbbbbbb" in md.read_text()


def test_report_says_when_it_was_measured_and_that_it_was_reused():
    doc = {"model_id": "m", "backend": "trainium", **_doc()}
    text = report.render(doc)
    assert "**Measured:** 2026-10-01T12:00:00Z on trn2.3xlarge at `aaaaaaaaaaaa`" in text
    assert "REUSED" not in text
    rc.mark_reused(doc, now=_days(5), platform=PLATFORM, code=CODE)
    assert "**REUSED:** these numbers were NOT re-measured" in report.render(doc)
    assert "Measured" not in report.render({"model_id": "m", "backend": "x"})  # legacy file


def test_status_rows(tmp_path):
    (tmp_path / "fresh.json").write_text(json.dumps(_doc()))
    (tmp_path / "old.json").write_text(json.dumps(_doc(measured_at=_days(-40))))
    (tmp_path / "legacy.json").write_text(json.dumps({"status": "ok"}))
    rows = dict(rc.status_rows(tmp_path, now=_days(2)))
    assert rows["fresh"].startswith("fresh (2.0 days old, expires in 28.0 days; trn2.3xlarge")
    assert rows["old"].startswith("EXPIRED (42.0 days old")
    assert rows["legacy"].startswith("no fingerprint")


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
def bench_env(tmp_path, monkeypatch):
    from benchmark import models

    FakeAdapter.generates = 0
    FakeAdapter.toolchain_versions = {"neuronx-cc": "2.21.0", "torch-neuronx": "2.9.0"}
    clock = {"now": T0}
    monkeypatch.setattr(models, "_RESULTS_ROOT", str(tmp_path))
    monkeypatch.setattr(models, "DEVICE", "trn2")
    monkeypatch.setattr(bench, "_make_adapter", lambda name: FakeAdapter())
    monkeypatch.setattr(rc, "instance_type", lambda: "trn2.3xlarge")
    monkeypatch.setattr(rc, "system_sdk", lambda: {"aws-neuronx-runtime-lib": "2.28.0"})
    monkeypatch.setattr(rc, "code_identity", lambda repo=None: dict(CODE))
    monkeypatch.setattr(rc, "utc_now", lambda: clock["now"])

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["bench", "--model", "flux_1_dev", *extra])
        before = FakeAdapter.generates
        assert bench.main() == 0
        return FakeAdapter.generates - before

    run.clock = clock
    run.json = tmp_path / "trn2" / "flux_1_dev.json"
    run.md = tmp_path / "trn2" / "flux_1_dev.md"
    return run


def test_bench_measures_once_then_reuses_on_the_next_instance(bench_env, capsys):
    assert bench_env() == 1  # measured: the cold generate
    doc = json.loads(bench_env.json.read_text())
    assert doc["compile_seconds"] == 1484.0
    assert doc["cache"]["measured_at"] == "2026-10-01T12:00:00Z"
    assert doc["cache"]["fingerprint"]["platform"]["sdk"]["aws-neuronx-runtime-lib"] == "2.28.0"
    assert "REUSED" not in bench_env.md.read_text()

    bench_env.clock["now"] = _days(5)
    assert bench_env() == 0  # nothing ran
    out = capsys.readouterr().out
    assert "[cache] flux_1_dev: REUSE -- measured 2026-10-01T12:00:00Z on trn2.3xlarge" in out
    doc = json.loads(bench_env.json.read_text())
    assert doc["compile_seconds"] == 1484.0 and doc["cache"]["last_reused"]["at"] == rc.iso(_days(5))
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
    doc = json.loads(bench_env.json.read_text())
    assert doc["cache"]["measured_at"] == rc.iso(_days(31))
    assert "last_reused" not in doc["cache"]  # a re-measurement starts a fresh entry


def test_bench_plan_runs_and_writes_nothing(bench_env, capsys):
    assert bench_env("--plan") == 0
    assert not bench_env.json.exists()
    assert "[cache] flux_1_dev: RUN -- no committed result" in capsys.readouterr().out
    bench_env()
    before = bench_env.json.read_bytes()
    assert bench_env("--plan") == 0
    assert bench_env.json.read_bytes() == before  # a plan does not mark anything reused
    assert "REUSE -- measured" in capsys.readouterr().out


def test_a_failed_run_is_not_reused(bench_env, monkeypatch, capsys):
    compiles = []

    def broken(self, spec):
        compiles.append(spec)
        raise RuntimeError("compile failed")

    monkeypatch.setattr(FakeAdapter, "compile", broken)
    bench_env()
    assert json.loads(bench_env.json.read_text())["status"] == "failed"
    bench_env()
    assert len(compiles) == 2  # attempted again, not served from the failed file
    assert "RUN -- committed result's status is 'failed'" in capsys.readouterr().out


# ------------------------------------------------------------------------------ neuron job


def _assembled(tmp_path, *, fp):
    records = {**_records(5.0), **_records(1.0)}
    _write_records(tmp_path / "rec", records)
    out_json, out_md = tmp_path / "wan.json", tmp_path / "wan.md"
    doc = cw.assemble(cw.load_records(tmp_path / "rec"), adapter=None, out_json=out_json,
                      out_md=out_md, fingerprint=fp)
    return doc, out_json, out_md


def test_assemble_records_the_cache_block_at_the_last_launch_time(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "code_identity", lambda repo=None: dict(CODE))
    doc, out_json, out_md = _assembled(tmp_path, fp=_fp())
    records = cw.load_records(tmp_path / "rec")
    last = max(r["run"]["end_epoch"] for r in records.values() if r["run"].get("end_epoch"))
    assert doc["cache"]["measured_at"] == rc.iso(
        datetime.datetime.fromtimestamp(last, datetime.timezone.utc))
    assert json.loads(out_json.read_text())["cache"]["key"] == rc.key_of(_fp())
    text = out_md.read_text()
    assert text.startswith("# Benchmark — Wan2.1")
    assert "\n\n**Measured:** " in text and "**Status:**" in text
    # no fingerprint (--assemble-only): the old report, byte for byte
    plain = cw.assemble(records, adapter=None, out_json=tmp_path / "p.json",
                        out_md=tmp_path / "p.md")
    assert "cache" not in plain and "**Measured:**" not in (tmp_path / "p.md").read_text()


def test_neuron_job_reuses_the_committed_result_without_launching(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rc, "code_identity", lambda repo=None: dict(CODE))
    monkeypatch.setattr(rc, "platform_identity", lambda adapter, **kw: dict(PLATFORM))
    monkeypatch.setattr(cw, "sudo_available", lambda: False)

    def no_launch(*a, **kw):
        raise AssertionError("a launch ran although the result was reusable")

    monkeypatch.setattr(cw, "run_launch", no_launch)
    monkeypatch.setattr(cw, "host_info", no_launch)
    argv = ["--record-dir", str(tmp_path / "rec"), "--true-cold", "no"]
    args = argparse.Namespace(
        guidance=list(cw.GUIDANCES), iters=3, warm_launches=3, cli_warm=3, no_w4_check=False,
        steps=None)
    fp = rc.fingerprint(rc.config_identity(MATRIX[cw.SLUG], backend="neuron",
                                           run_options=cw.run_options(args, true_cold=False)),
                        PLATFORM)
    doc, out_json, out_md = _assembled(tmp_path, fp=fp)
    measured = rc.parse_iso(doc["cache"]["measured_at"])
    monkeypatch.setattr(rc, "utc_now", lambda: measured + datetime.timedelta(days=2))

    assert cw.main([*argv, "--out-json", str(out_json), "--out-md", str(out_md)]) == 0
    assert "[cache] REUSE -- measured" in capsys.readouterr().out
    reused = json.loads(out_json.read_text())
    assert reused["cache"]["last_reused"]["instance_type"] == "trn2.3xlarge"
    assert reused["results_by_guidance"] == doc["results_by_guidance"]
    assert "**REUSED:**" in out_md.read_text() and "## Headline" in out_md.read_text()

    # --true-cold yes is another measurement: the job runs (here: its first launch raises)
    with pytest.raises(AssertionError, match="a launch ran"):
        cw.main([*argv[:-1], "yes", "--out-json", str(out_json), "--out-md", str(out_md)])
    # and so is --force
    with pytest.raises(AssertionError, match="a launch ran"):
        cw.main([*argv, "--force", "--out-json", str(out_json), "--out-md", str(out_md)])
