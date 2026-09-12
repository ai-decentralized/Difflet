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
@pytest.mark.parametrize("config", ["tp4", "tp2cp2", "tp4sp", "tp2cfg", "tp4sdpa"])
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
    assert ns.work_dir == str(work) and ns.cache_dir == str(tmp_path)
    assert float(ns.guidance_scale) == float(cfg.guidance_scale)


def test_every_campaign_model_has_a_builder():
    for slug in ("flux_1_dev", "ltx_2", "wan_2_1", "qwen_image", "hunyuan_video"):
        assert slug in sr._BUILDERS
