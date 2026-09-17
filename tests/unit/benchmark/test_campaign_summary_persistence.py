"""campaign_summary keeps its log/file-derived evidence in the cell JSON so the
report regenerates faithfully on a host that has neither the output files nor
the realloop logs (the 2026-09-17 tp4tcad host vs the 2026-09-13 TeaCache
rows)."""
from __future__ import annotations

import json
import os

import pytest

from benchmark import campaign_summary as cs


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A scratch repo root: benchmark/trn2/<cell>.json without logs or outputs."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "benchmark" / "trn2" / "logs").mkdir(parents=True)
    return tmp_path


def _write(repo, slug, label, d):
    from benchmark.models import resolve
    p = repo / "benchmark" / "trn2" / f"{resolve(slug, label).config_slug}.json"
    p.write_text(json.dumps(d))
    return p


def test_parity_reads_the_persisted_record_when_outputs_are_absent(repo):
    d = {"device_slug": "trn2", "output_vs_tp4": {"identical": False, "psnr_db": 41.3,
                                                  "source": "campaign host"}}
    assert cs._parity("flux_1_dev", "tp4tc2", d) == "41.3 dB"
    assert cs._parity("flux_1_dev", "tp4tc2", {"device_slug": "trn2"}) == "—"
    assert cs._parity("flux_1_dev", "tp4tc2",
                      {"output_vs_tp4": {"identical": True}}) == "**identical (no-op)**"


def test_parity_persists_a_fresh_comparison_into_the_cell_json(repo, monkeypatch):
    from PIL import Image
    logs = repo / "benchmark" / "trn2" / "logs"
    Image.new("RGB", (8, 8), (10, 10, 10)).save(logs / "flux_1_dev_out.png")
    Image.new("RGB", (8, 8), (12, 10, 10)).save(logs / "flux_1_dev_tp4tc2_out.png")
    jp = _write(repo, "flux_1_dev", "tp4tc2", {"device_slug": "trn2"})
    d = json.loads(jp.read_text())
    s = cs._parity("flux_1_dev", "tp4tc2", d)
    assert s.endswith("dB")
    persisted = json.loads(jp.read_text())["output_vs_tp4"]
    assert persisted["identical"] is False and persisted["psnr_db"] > 0
    assert "on this host" in persisted["source"]
    # with the files gone the persisted value renders the same
    os.remove(logs / "flux_1_dev_tp4tc2_out.png")
    assert cs._parity("flux_1_dev", "tp4tc2", persisted and json.loads(jp.read_text())) == s


def test_teacache_skips_prefer_the_persisted_stats_line_over_the_call_count(repo):
    d = {"device_slug": "trn2",
         "teacache": {"skipped_steps_by_calls": 9, "skipped_steps_stats_line": 9}}
    assert cs._teacache_skips("flux_1_dev", "tp4tc2", d) == (9, "stats line")
    d = {"device_slug": "trn2", "teacache": {"skipped_steps_by_calls": 5}}
    assert cs._teacache_skips("qwen_image", "tp4tc2", d) == (5, "DiT-call count")
    assert cs._teacache_skips("qwen_image", "tp4tc2", {"device_slug": "trn2"}) == (None, "—")


def test_teacache_skips_persist_the_stats_line_from_the_realloop_log(repo):
    log = repo / "benchmark" / "trn2" / "logs" / "tp4tcad"
    log.mkdir()
    (log / "wan_2_1_realloop.log").write_text(
        "[teacache] stats: {'full_steps': 15, 'skipped_steps': 5, 'probe_calls': 0}\n")
    jp = _write(repo, "wan_2_1", "tp4tcad",
                {"device_slug": "trn2", "teacache": {"skipped_steps_by_calls": 5}})
    d = json.loads(jp.read_text())
    assert cs._teacache_skips("wan_2_1", "tp4tcad", d) == (5, "stats line")
    assert json.loads(jp.read_text())["teacache"]["skipped_steps_stats_line"] == 5
    os.remove(log / "wan_2_1_realloop.log")
    assert cs._teacache_skips("wan_2_1", "tp4tcad", json.loads(jp.read_text())) == (5, "stats line")
