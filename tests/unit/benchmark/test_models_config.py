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


def test_attention_impl_config():
    """tp4sdpa = the tp4 topology with --attention-impl sdpa; the flag and the
    parallel record carry it only when it is not the megakernel default, so the
    pre-existing tp4 files (no attention_impl key) still match their label."""
    cfg = resolve("flux_1_dev", "tp4sdpa")
    assert cfg.tp == 4 and cfg.cp == 1 and cfg.attention_impl == "sdpa"
    assert cfg.parallel_flags()[-2:] == ["--attention-impl", "sdpa"]
    assert cfg.parallel_dict()["attention_impl"] == "sdpa"
    assert cfg.config_slug == "flux_1_dev_tp4sdpa"
    base = resolve("flux_1_dev", "tp4")
    assert "--attention-impl" not in base.parallel_flags()
    assert "attention_impl" not in base.parallel_dict()


def test_cfg_baseline_config():
    cfg = resolve("wan_2_1", "tp4cfg2")
    assert cfg.tp == 4 and not cfg.cfg_parallel and cfg.guidance_scale == 2.0
    assert cfg.parallel_flags() == ["--tp-degree", "4", "--cp-degree", "1"]
    assert cfg.config_slug == "wan_2_1_tp4cfg2"
    for m in ("flux_1_dev", "qwen_image", "hunyuan_video"):
        assert (m, "tp4cfg2") in UNSUPPORTED
    for m in ("wan_2_1", "ltx_2"):
        assert (m, "tp4cfg2") not in UNSUPPORTED


def test_teacache_configs():
    """Runtime-only TeaCache overlays on the tp4 artifact: the flags go to
    generate only, the topology record is unchanged (same artifact), the JSON
    carries a teacache record."""
    tc = resolve("flux_1_dev", "tp4tc2")
    od = resolve("wan_2_1", "tp4tcod")
    assert tc.teacache_flags() == ["--teacache-cadence", "2"]
    assert od.teacache_flags() == ["--teacache-online-delta", "0.6"]
    assert tc.parallel_flags() == resolve("flux_1_dev", "tp4").parallel_flags()
    assert tc.parallel_dict() == resolve("flux_1_dev", "tp4").parallel_dict()
    assert tc.teacache_dict()["mode"] == "fixed_cadence" and tc.teacache_dict()["cadence"] == 2
    assert od.teacache_dict()["mode"] == "online_delta" and od.teacache_dict()["online_delta_alpha"] == 0.6
    assert resolve("flux_1_dev", "tp4").teacache_dict() is None
    assert tc.config_slug == "flux_1_dev_tp4tc2" and od.config_slug == "wan_2_1_tp4tcod"


def test_adaptive_teacache_config():
    """tp4tcad = calibrated adaptive at cadence 2's skip budget: the per-model
    target and calibration path go to generate AND compile (the probe NEFF is
    part of the artifact), the topology record is unchanged, and the result
    record names the mode, target and calibration."""
    from benchmark.models import (adaptive_target_speedup, cadence2_skips,
                                  teacache_calibration_path)
    assert cadence2_skips(28) == 9 and cadence2_skips(20) == 5
    assert adaptive_target_speedup(28) == 1.474 and adaptive_target_speedup(20) == 1.333
    fx = resolve("flux_1_dev", "tp4tcad")
    assert fx.teacache_speedup == 1.474
    assert fx.teacache_calibration == teacache_calibration_path("flux_1_dev")
    assert fx.teacache_calibration.endswith("teacache_calib/flux_1_dev_tp4tcad.json")
    assert os.path.isabs(fx.teacache_calibration)
    assert fx.compile_teacache_flags() == ["--teacache-speedup", "1.474",
                                           "--teacache-calibration", fx.teacache_calibration]
    assert fx.teacache_flags() == fx.compile_teacache_flags()
    assert fx.parallel_flags() == resolve("flux_1_dev", "tp4").parallel_flags()
    assert fx.parallel_dict() == resolve("flux_1_dev", "tp4").parallel_dict()
    assert fx.config_slug == "flux_1_dev_tp4tcad"
    d = fx.teacache_dict()
    assert d["mode"] == "adaptive" and d["target_speedup"] == 1.474
    assert d["cadence"] is None and d["online_delta_alpha"] is None
    assert d["calibration"] == fx.teacache_calibration
    assert d["warmup_steps"] == 5 and d["cooldown_steps"] == 5
    assert resolve("wan_2_1", "tp4tcad").teacache_speedup == 1.333
    # the probe-free overlays never touch compile
    assert resolve("flux_1_dev", "tp4tc2").compile_teacache_flags() == []
    assert resolve("flux_1_dev", "tp4tcod").teacache_flags() == ["--teacache-online-delta", "0.6"]
    # every campaign model is wired for the calibrated mode (Wan / LTX-2 via the
    # host signal), so there is no by-design N/A cell
    for m in CAMPAIGN:
        assert (m, "tp4tcad") not in UNSUPPORTED


def test_adaptive_teacache_record_carries_the_calibration_fit(tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    calib = tmp_path / "c.json"
    calib.write_text(json.dumps({"fit_r2": 0.83, "signal_pearson": 0.91, "threshold": 0.27,
                                 "accumulate": True, "poly_coef": [0, 1, 2, 3, 4],
                                 "n_samples": 81}))
    cfg = replace(resolve("flux_1_dev", "tp4tcad"), teacache_calibration=str(calib))
    d = cfg.teacache_dict()
    assert d["fit_r2"] == 0.83 and d["signal_pearson"] == 0.91 and d["threshold"] == 0.27
    assert d["accumulate"] is True and d["poly_degree"] == 4 and d["n_samples"] == 81
    missing = replace(cfg, teacache_calibration=str(tmp_path / "absent.json")).teacache_dict()
    assert missing["mode"] == "adaptive" and "fit_r2" not in missing


def test_online_delta_sweep_labels_are_tp4tcod_at_other_alphas():
    """One cell per alpha, generate-only flag, same tp4 artifact / topology record;
    the sweep labels are their own experiment (is_sweep_label), tp4tcod is not."""
    from benchmark.models import ONLINE_DELTA_SWEEP, is_sweep_label
    assert ONLINE_DELTA_SWEEP == {"tp4tcod02": 0.2, "tp4tcod03": 0.3, "tp4tcod04": 0.4,
                                  "tp4tcod05": 0.5, "tp4tcod06": 0.6, "tp4tcod08": 0.8}
    tp4 = resolve("hunyuan_video", "tp4")
    for label, a in ONLINE_DELTA_SWEEP.items():
        cfg = resolve("hunyuan_video", label)
        assert cfg.teacache_flags() == ["--teacache-online-delta", str(a)]
        assert cfg.compile_teacache_flags() == []
        assert cfg.parallel_dict() == tp4.parallel_dict()
        assert cfg.teacache_dict()["mode"] == "online_delta"
        assert cfg.teacache_dict()["online_delta_alpha"] == a
        assert cfg.config_slug == f"hunyuan_video_{label}"
        assert is_sweep_label(label) and str(a) in cfg.config_label
    assert not is_sweep_label("tp4tcod") and not is_sweep_label("tp4tc2")


def test_configs_are_the_verify_cli_labels_sized_to_four_cores():
    from benchmark.models import ONLINE_DELTA_SWEEP
    assert set(CONFIGS) == {"tp4", "tp2cp2", "tp4sp", "tp2cfg", "tp4sdpa", "tp4cfg2",
                            "tp4tc2", "tp4tcod", "tp4tcad", *ONLINE_DELTA_SWEEP}
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
    topologies = ["tp4", "tp2cp2", "tp4sp", "tp2cfg"]
    cells = {(m, c) for m in CAMPAIGN for c in topologies if (m, c) in UNSUPPORTED}
    assert cells == {
        ("flux_1_dev", "tp2cfg"), ("qwen_image", "tp2cfg"), ("hunyuan_video", "tp2cfg"),
        ("ltx_2", "tp2cp2"), ("ltx_2", "tp4sp"),
    }
    topologies = ["tp4", "tp2cp2", "tp4sp", "tp2cfg"]
    assert sum(1 for m in CAMPAIGN for c in topologies if (m, c) not in UNSUPPORTED) == 15
    # the attention-impl variant runs the tp4 topology for every model
    assert all((m, "tp4sdpa") not in UNSUPPORTED for m in CAMPAIGN)


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


def test_cell_status_teacache_cell_uses_the_call_count(tmp_path, monkeypatch):
    """A TeaCache cell makes fewer DiT calls than steps; completeness is judged
    against the recorded call count, not steps - 1."""
    monkeypatch.setattr("benchmark.models._RESULTS_ROOT", str(tmp_path))
    cfg = resolve("flux_1_dev", "tp4tc2")
    os.makedirs(os.path.dirname(json_path(cfg.config_slug)), exist_ok=True)
    d = {"status": "ok", "config": "tp4tc2", "parallel": cfg.parallel_dict(),
         "compile_seconds": 22.0, "e2e_cold_seconds": 306.0, "e2e_warm": {"mean": 38.7},
         "step_latency": {"mean": 0.27, "n": 18}, "dit_calls": 19,
         "teacache": cfg.teacache_dict()}
    with open(json_path(cfg.config_slug), "w") as fh:
        json.dump(d, fh)
    assert cell_status.missing("flux_1_dev", "tp4tc2") == []


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
