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


def test_stats_line_with_trace_lists_still_parses(repo):
    """The instrumented stats line carries list fields; the first-brace regex
    + literal_eval must still read skipped_steps from it."""
    log = repo / "benchmark" / "trn2" / "logs" / "tp4tcod03"
    log.mkdir()
    (log / "flux_1_dev_realloop.log").write_text(
        "[teacache] stats: {'full_steps': 25, 'skipped_steps': 3, 'probe_calls': 0, "
        "'last_delta_estimate': 0.01, 'cache_initialized': True, 'online_delta_alpha': 0.3, "
        "'baseline_delta': 0.5, 'skipped_step_indices': [9, 13, 17], "
        "'delta_trace': [[1, 0.5], [2, 0.3]]}\n")
    jp = _write(repo, "flux_1_dev", "tp4tcod03", {"device_slug": "trn2", "teacache": {}})
    assert cs._teacache_skips("flux_1_dev", "tp4tcod03", json.loads(jp.read_text())) == (3, "stats line")


def test_alpha_sweep_table_reads_indices_and_what_if_from_the_trace(repo):
    # a 20-step trace where deltas fall from 0.5 (baseline) to 0.1: at alpha 0.3
    # step 5 sees step 4's delta 0.23 (ratio 0.46, run), step 6 sees step 5's
    # 0.14 (ratio 0.28 < 0.3 -> the first skip); the knee is 0.1/0.5 = 0.20
    trace = [[s, round(0.5 - 0.09 * (s - 1), 3) if s < 6 else 0.1] for s in range(1, 20)]
    _write(repo, "wan_2_1", "tp4", {"device_slug": "trn2", "steps": 20, "loop_step_ms": 575.0,
                                    "e2e_warm": {"mean": 85.0}})
    _write(repo, "wan_2_1", "tp4tcod03",
           {"device_slug": "trn2", "steps": 20, "loop_step_ms": 500.0, "e2e_warm": {"mean": 82.0},
            "teacache": {"mode": "online_delta", "online_delta_alpha": 0.3, "warmup_steps": 5,
                         "cooldown_steps": 5, "skipped_steps_by_calls": 3,
                         "stats": {"skipped_step_indices": [5, 7, 9], "delta_trace": trace}},
            "output_vs_tp4": {"identical": False, "psnr_db": 38.2, "ssim": 0.97}})
    md = "\n".join(cs.alpha_sweep_table(["tp4tcod03", "tp4tcod02"], models=["wan_2_1"]))
    row = next(l for l in md.splitlines() if "0.3 (tp4tcod03)" in l)
    assert "**3/20** (DiT-call count)" in row and "| 5, 7, 9 |" in row
    assert "575 → **500** (1.15×)" in row and "38.2 dB, SSIM 0.970" in row
    assert "first skip @ step 6" in row and "knee α ≈ 0.20" in row and "(from tp4tcod03)" in row
    assert "0.2 (tp4tcod02) | not measured" in md
    assert cs._trace_knee([], 0.3, 5, 20, 5) == (None, None)
