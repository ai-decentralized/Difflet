"""step_realloop staged builders: the orchestrator argv they hand to the stage
parser must carry the exact topology of the --config label (a dropped --sp or
--cp-mode would time the wrong artifact -- the class of bug that made the
legacy step_latency timer silently mis-measure)."""
from __future__ import annotations

import pytest

pytest.importorskip("difflet.cli.stage")

from benchmark import step_realloop as sr
from benchmark.models import resolve


@pytest.mark.parametrize("orch_mod, model_slug, stage", [
    ("difflet.cli.orchestrators.wan", "wan_2_1", "transformer"),
    ("difflet.cli.orchestrators.qwen_image", "qwen_image", "generate"),
    ("difflet.cli.orchestrators.hunyuan_video", "hunyuan_video", "generate"),
])
@pytest.mark.parametrize("config", ["tp4", "tp2cp2", "tp4sp", "tp2cfg", "tp4sdpa", "tp4tc2", "tp4tcod",
                                    "tp4tcad"])
def test_stage_namespace_carries_the_config_topology(tmp_path, orch_mod, model_slug,
                                                     stage, config):
    import importlib
    mod = importlib.import_module(orch_mod)
    cls = next(v for k, v in vars(mod).items() if k.endswith("Orchestrator")
               and isinstance(v, type) and k != "ModelOrchestrator")
    cfg = resolve(model_slug, config)
    work = tmp_path / "work"
    work.mkdir()
    orch = cls(sr._staged_namespace(cfg, tmp_path, work, work / "out.pt"))
    argv = orch._shared_cli_args(stage_mode="generate", work_dir=str(work))
    ns = sr._stage_ns(cfg.model_id, stage, argv)
    assert ns.tp_degree == cfg.tp and ns.cp_degree == cfg.cp
    assert ns.cp_mode == cfg.cp_mode
    assert bool(ns.sp_enabled) == cfg.sp
    if config == "tp2cfg":
        # only the true-CFG orchestrator (wan) threads --cfg-parallel; the
        # distilled ones are N/A cells that the driver never runs
        assert bool(getattr(ns, "cfg_parallel", False)) == (model_slug == "wan_2_1")
    assert ns.steps == cfg.steps and ns.stage_mode == "generate"
    assert getattr(ns, "attention_impl", "megakernel") == cfg.attention_impl
    # TeaCache flags reach the stage argv (runtime-only; same artifact)
    assert getattr(ns, "teacache_cadence", None) == cfg.teacache_cadence
    assert getattr(ns, "teacache_online_delta", None) == cfg.teacache_online_delta
    assert sr._teacache_app_kwargs(cfg) == (
        {"teacache_cadence": 2} if config == "tp4tc2"
        else {"teacache_online_delta_alpha": 0.6} if config == "tp4tcod"
        else {"teacache_speedup": cfg.teacache_speedup,
              "teacache_calibration_path": cfg.teacache_calibration} if config == "tp4tcad"
        else {})
    # calibrated adaptive: the target and calibration reach the stage argv too
    # (the staged orchestrators forward them; the stage parser accepts them)
    assert getattr(ns, "teacache_speedup", None) == cfg.teacache_speedup
    assert getattr(ns, "teacache_calibration", None) == cfg.teacache_calibration
    assert ns.work_dir == str(work) and ns.cache_dir == str(tmp_path)
    assert float(ns.guidance_scale) == float(cfg.guidance_scale)


def test_ltx2_adaptive_kwargs_keep_the_tp4_artifact_identity():
    """LTX-2 has no probe NEFF: the in-process loader passes only the calibration
    path (like difflet/cli/orchestrators/ltx_2.py), never teacache_speedup, which
    would flip DiffletPipeline's cache key to a probe identity."""
    cfg = resolve("ltx_2", "tp4tcad")
    assert sr._teacache_app_kwargs(cfg) == {"teacache_calibration_path": cfg.teacache_calibration}
    fx = resolve("flux_1_dev", "tp4tcad")
    assert sr._teacache_app_kwargs(fx) == {"teacache_speedup": fx.teacache_speedup,
                                           "teacache_calibration_path": fx.teacache_calibration}


def test_every_campaign_model_has_a_builder():
    for slug in ("flux_1_dev", "ltx_2", "wan_2_1", "qwen_image", "hunyuan_video"):
        assert slug in sr._BUILDERS
