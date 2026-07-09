from __future__ import annotations

import pytest

import difflet.cli.stage as stage


# ----------------------------------------------------------- _load_orchestrator_class

def test_load_unknown_orchestrator_exits():
    with pytest.raises(SystemExit) as exc:
        stage._load_orchestrator_class("not-a-real-model")
    assert "unknown orchestrator" in str(exc.value)


def test_load_known_orchestrator_returns_class():
    from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
    cls = stage._load_orchestrator_class("Qwen/Qwen-Image")
    assert cls is QwenImageOrchestrator


@pytest.mark.parametrize("model_id", list(stage._ORCHESTRATOR_MAP))
def test_load_every_mapped_orchestrator(model_id):
    cls = stage._load_orchestrator_class(model_id)
    assert isinstance(cls, type)
    # The dotted path's final component matches the class name.
    assert cls.__name__ == stage._ORCHESTRATOR_MAP[model_id].rsplit(".", 1)[1]


# ----------------------------------------------------------- _build_stage_parser

def test_parser_defaults():
    p = stage._build_stage_parser()
    args, _ = p.parse_known_args(["--orchestrator", "X", "--stage", "transformer"])
    assert args.orchestrator == "X"
    assert args.stage == "transformer"
    assert args.stage_mode == "generate"
    assert args.cp_degree == 1
    assert args.cp_mode == "gather_kv"
    assert args.cfg_parallel is False
    assert args.seed == 42
    assert args.model_id is None


def test_parser_full_args_and_ignores_unknown():
    p = stage._build_stage_parser()
    args, extra = p.parse_known_args([
        "--orchestrator", "Qwen/Qwen-Image", "--stage", "vae",
        "--stage-mode", "compile", "--model-id", "Qwen/Qwen-Image",
        "--tp-degree", "8", "--cp-degree", "2", "--cp-mode", "ring",
        "--cfg-parallel", "--height", "512", "--width", "768",
        "--num-frames", "9", "--prompt", "a cat", "--output", "/tmp/o.png",
        "--steps", "3", "--guidance-scale", "2.5", "--seed", "7",
        "--work-dir", "/tmp/w", "--cache-dir", "/tmp/c",
        "--teacache-cadence", "2", "--teacache-online-delta", "0.5",
        "--teacache-speedup", "1.4", "--teacache-calibration", "/tmp/cal.json",
        "--some-unknown-flag", "z",
    ])
    assert args.stage_mode == "compile"
    assert args.tp_degree == 8
    assert args.cp_degree == 2
    assert args.cp_mode == "ring"
    assert args.cfg_parallel is True
    assert args.guidance_scale == 2.5
    assert args.teacache_calibration == "/tmp/cal.json"
    assert "--some-unknown-flag" in extra


# ----------------------------------------------------------- main

def test_main_dispatches_to_run_stage_internal(monkeypatch):
    recorded = {}

    class FakeOrch:
        def __init__(self, args):
            recorded["init_args"] = args

        def _run_stage_internal(self, stage_name, args):
            recorded["stage"] = stage_name
            recorded["run_args"] = args

    monkeypatch.setattr(stage, "_load_orchestrator_class", lambda name: FakeOrch)
    rc = stage.main([
        "--orchestrator", "anything", "--stage", "transformer",
        "--model-id", "m",
    ])
    assert rc == 0
    assert recorded["stage"] == "transformer"
    assert recorded["init_args"] is recorded["run_args"]
    assert recorded["run_args"].model_id == "m"


def test_main_propagates_unknown_orchestrator_exit(monkeypatch):
    # _load_orchestrator_class is the real one -> unknown id -> SystemExit.
    with pytest.raises(SystemExit):
        stage.main(["--orchestrator", "bogus", "--stage", "transformer"])


# ------------------------------------------- forwarded-flag coverage (drift guard)

def test_parser_parses_sp_into_sp_enabled():
    p = stage._build_stage_parser()
    args, _ = p.parse_known_args(["--orchestrator", "X", "--stage", "transformer", "--sp"])
    assert args.sp_enabled is True
    args, _ = p.parse_known_args(["--orchestrator", "X", "--stage", "transformer"])
    assert args.sp_enabled is False


@pytest.mark.parametrize(
    "orch_module, orch_cls, model_id",
    [
        ("difflet.cli.orchestrators.wan", "WanOrchestrator",
         "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
        ("difflet.cli.orchestrators.hunyuan_video", "HunyuanVideoOrchestrator",
         "hunyuanvideo-community/HunyuanVideo"),
        ("difflet.cli.orchestrators.qwen_image", "QwenImageOrchestrator",
         "Qwen/Qwen-Image"),
    ],
)
def test_stage_parser_consumes_every_forwarded_flag(orch_module, orch_cls, model_id):
    """Every flag an orchestrator forwards via _shared_cli_args must be consumed
    by the stage parser — parse_known_args silently drops unknown flags, which
    turned --sp into a no-op (SP cells silently ran the dense graph)."""
    import argparse
    import importlib

    cls = getattr(importlib.import_module(orch_module), orch_cls)
    ns = argparse.Namespace(
        model_id=model_id, tp_degree=4, cp_degree=2, cp_mode="ring",
        cfg_parallel=True, sp_enabled=True,
        height=64, width=96, num_frames=9,
        steps=2, guidance_scale=2.0, seed=7,
        prompt="a cat", output="/tmp/o.mp4",
        cache_dir="/tmp/c", work_dir=None, keep_work_dir=False,
        force=False, revision=None,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    forwarded = cls(ns)._shared_cli_args("compile", work_dir="/tmp/w")

    parser = stage._build_stage_parser()
    _, extra = parser.parse_known_args(
        ["--orchestrator", model_id, "--stage", "transformer", *forwarded]
    )
    assert extra == [], f"stage parser silently drops forwarded flags: {extra}"
