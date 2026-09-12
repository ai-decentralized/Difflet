"""benchmark.models: parallel-config labels, result-file naming, N/A table.

The N/A table (UNSUPPORTED) is hand-written so benchmark/ stays stdlib-only;
these tests pin it to the difflet registry / CLI gates so it cannot drift.
"""
from __future__ import annotations

import json
import os

import pytest

from benchmark import cell_status, mark_na
from benchmark.models import (CONFIGS, MATRIX, UNSUPPORTED, BenchConfig, json_path,
                              report_path, resolve)

CAMPAIGN = ["flux_1_dev", "qwen_image", "ltx_2", "hunyuan_video", "wan_2_1"]


def test_configs_are_the_verify_cli_labels_sized_to_four_cores():
    assert set(CONFIGS) == {"tp4", "tp2cp2", "tp4sp", "tp2cfg"}
    for label in CONFIGS:
        cfg = resolve("flux_1_dev", label)
        world = cfg.tp * cfg.cp * (2 if cfg.cfg_parallel else 1)
        assert world == 4, (label, world)


def test_parallel_flags_per_config():
    flags = {label: resolve("wan_2_1", label).parallel_flags() for label in CONFIGS}
    assert flags["tp4"] == ["--tp-degree", "4", "--cp-degree", "1"]
    assert flags["tp2cp2"] == ["--tp-degree", "2", "--cp-degree", "2", "--cp-mode", "ulysses"]
    assert flags["tp4sp"] == ["--tp-degree", "4", "--cp-degree", "1", "--sp"]
    assert flags["tp2cfg"] == ["--tp-degree", "2", "--cp-degree", "1", "--cfg-parallel"]


def test_tp2cfg_runs_true_cfg_at_guidance_two_only_there():
    assert resolve("wan_2_1", "tp2cfg").guidance_scale == 2.0
    assert resolve("ltx_2", "tp2cfg").guidance_scale == 2.0
    for label in ("tp4", "tp2cp2", "tp4sp"):
        assert resolve("wan_2_1", label).guidance_scale == MATRIX["wan_2_1"].guidance_scale


def test_resolve_leaves_the_matrix_untouched():
    before = MATRIX["flux_1_dev"]
    cfg = resolve("flux_1_dev", "tp2cp2")
    assert cfg is not before and cfg.cp == 2
    assert MATRIX["flux_1_dev"].cp == 1 and MATRIX["flux_1_dev"].slug == ""


def test_config_slug_keeps_tp4_history_files():
    assert resolve("flux_1_dev", "tp4").config_slug == "flux_1_dev"
    assert resolve("flux_1_dev", "tp2cp2").config_slug == "flux_1_dev_tp2cp2"
    assert json_path("flux_1_dev_tp2cp2").endswith("/flux_1_dev_tp2cp2.json")
    assert report_path("flux_1_dev").endswith("/flux_1_dev.md")
    # never collides with the pre-existing flux_<label>.json sweep files
    assert not resolve("flux_1_dev", "tp2cp2").config_slug.startswith("flux_tp")


def test_resolve_rejects_unknown():
    with pytest.raises(KeyError):
        resolve("nope", "tp4")
    with pytest.raises(KeyError):
        resolve("flux_1_dev", "tp8")


def test_unsupported_cells_are_exactly_the_documented_five_for_the_campaign():
    cells = {(m, c) for m in CAMPAIGN for c in CONFIGS if (m, c) in UNSUPPORTED}
    assert cells == {
        ("flux_1_dev", "tp2cfg"), ("qwen_image", "tp2cfg"), ("hunyuan_video", "tp2cfg"),
        ("ltx_2", "tp2cp2"), ("ltx_2", "tp4sp"),
    }
    assert sum(1 for m in CAMPAIGN for c in CONFIGS if (m, c) not in UNSUPPORTED) == 15


def test_unsupported_matches_registry_capabilities():
    """The hand-written N/A table must agree with what difflet itself gates."""
    registry = pytest.importorskip("difflet.registry")
    for slug in CAMPAIGN:
        cfg = MATRIX[slug]
        caps = registry.resolve_model(cfg.model_id, model_type=cfg.model_type).capabilities
        assert caps is not None, slug
        assert ((slug, "tp2cfg") in UNSUPPORTED) == (not caps.supports_cfg_parallel), slug
        assert ((slug, "tp2cp2") in UNSUPPORTED) == (not caps.supports_cp), slug
        assert ((slug, "tp4sp") in UNSUPPORTED) == (not caps.supports_sp), slug
        if caps.supports_cp:
            assert "ulysses" in caps.cp_modes, slug


def test_mark_na_and_cell_status_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark.models._RESULTS_ROOT", str(tmp_path))
    d = mark_na.skipped_record("ltx_2", "tp2cp2", UNSUPPORTED[("ltx_2", "tp2cp2")])
    assert d["status"] == "skipped" and d["config_slug"] == "ltx_2_tp2cp2"
    assert d["parallel"]["cp_mode"] == "ulysses"
    from benchmark import report
    md = report.render(d)   # a skipped record renders without a measured field
    assert "ltx_2" in md and "--config tp2cp2" in md
    os.makedirs(os.path.dirname(json_path("ltx_2_tp2cp2")), exist_ok=True)
    with open(json_path("ltx_2_tp2cp2"), "w") as fh:
        json.dump(d, fh)
    assert cell_status.missing("ltx_2", "tp2cp2") == []
    assert cell_status.missing("ltx_2", "tp4") == ["result file"]


def test_cell_status_ignores_pre_campaign_history(tmp_path, monkeypatch):
    """The historical tp4 <slug>.json (no `config` field) carries every metric
    but is not a measurement of this run -- the driver must re-run the cell."""
    monkeypatch.setattr("benchmark.models._RESULTS_ROOT", str(tmp_path))
    os.makedirs(os.path.dirname(json_path("flux_1_dev")), exist_ok=True)
    old = {"status": "ok", "parallel": {"tp_degree": 4, "cp_degree": 1},
           "compile_seconds": 1484.0, "e2e_cold_seconds": 321.0,
           "e2e_warm": {"mean": 35.0}, "step_latency": {"mean": 0.2676, "n": 27},
           "config_slug": "flux_1_dev"}
    with open(json_path("flux_1_dev"), "w") as fh:
        json.dump(old, fh)
    m = cell_status.missing("flux_1_dev", "tp4")
    assert len(m) == 1 and m[0].startswith("result file")


def test_cell_status_requires_all_four_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark.models._RESULTS_ROOT", str(tmp_path))
    cfg = resolve("flux_1_dev", "tp4sp")
    os.makedirs(os.path.dirname(json_path(cfg.config_slug)), exist_ok=True)
    d = {"status": "ok", "config": "tp4sp", "parallel": cfg.parallel_dict(),
         "compile_seconds": 100.0,
         "e2e_cold_seconds": 300.0, "e2e_warm": {"mean": 40.0},
         "step_latency": {"mean": 0.27, "n": 27}}
    with open(json_path(cfg.config_slug), "w") as fh:
        json.dump(d, fh)
    assert cell_status.missing("flux_1_dev", "tp4sp") == []
    d["step_latency"]["n"] = 5
    d["parallel"]["sp_enabled"] = False
    with open(json_path(cfg.config_slug), "w") as fh:
        json.dump(d, fh)
    m = cell_status.missing("flux_1_dev", "tp4sp")
    assert any(x.startswith("step_latency.n=5") for x in m)
    assert any(x.startswith("parallel mismatch") for x in m)
