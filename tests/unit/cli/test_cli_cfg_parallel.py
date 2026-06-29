"""CLI wiring for --cfg-parallel.

Pure argparse / config-construction tests (no Neuron, no torch import), so they
run under the default unit-test conftest.
"""
from __future__ import annotations

import argparse
import importlib

import pytest

from difflet.cli.main import _build_parser


def _cli():
    return importlib.import_module("difflet.cli.main")


# ---------------------------------------------------------------- parser

def test_cfg_parallel_defaults_false():
    parser = _build_parser()
    args = parser.parse_args(["compile", "--model-id", "x"])
    assert args.cfg_parallel is False


def test_cfg_parallel_store_true():
    parser = _build_parser()
    args = parser.parse_args(
        ["generate", "--model-id", "x", "--cfg-parallel",
         "--prompt", "p", "--output", "/tmp/o.mp4"]
    )
    assert args.cfg_parallel is True


# ---------------------------------------------------------------- main() guards

@pytest.mark.parametrize(
    "model_id",
    [
        "black-forest-labs/FLUX.1-dev",
        "hunyuanvideo-community/HunyuanVideo",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "Qwen/Qwen-Image",
    ],
)
def test_cfg_parallel_rejected_for_distilled_model(model_id, monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["compile", "--model-id", model_id, "--cfg-parallel"])
    assert exc.value.code == 1
    assert "guidance-distilled" in capsys.readouterr().err


def test_cfg_parallel_with_cp_degree_rejected(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "compile", "--model-id", "Lightricks/LTX-2",
            "--cfg-parallel", "--cp-degree", "2",
        ])
    assert exc.value.code == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_cfg_parallel_allowed_for_supported_model(monkeypatch):
    cli = _cli()
    calls = []

    class FakeOrch:
        def compile(self):
            calls.append("compile")

    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: FakeOrch())
    cli.main(["compile", "--model-id", "Lightricks/LTX-2", "--cfg-parallel"])
    assert calls == ["compile"]


# --------------------------------------------------- orchestrator threading

def _wan_args(**over):
    base = dict(
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        tp_degree=4, cp_degree=1, cp_mode="gather_kv", cfg_parallel=True,
        height=480, width=832, num_frames=9, steps=2, guidance_scale=5.0,
        seed=42, prompt="p", output="o.mp4", cache_dir=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_wan_compiled_dir_marks_cfg():
    from difflet.cli.orchestrators.wan import WanOrchestrator

    orch = WanOrchestrator(_wan_args())
    with_cfg = orch._stage_compiled_dir("transformer", _wan_args(cfg_parallel=True))
    without = orch._stage_compiled_dir("transformer", _wan_args(cfg_parallel=False))
    assert "cfg" in with_cfg.name
    assert "cfg" not in without.name
    assert with_cfg != without


def test_wan_shared_cli_args_forwards_flag():
    from difflet.cli.orchestrators.wan import WanOrchestrator

    on = WanOrchestrator(_wan_args(cfg_parallel=True))._shared_cli_args(stage_mode="compile")
    off = WanOrchestrator(_wan_args(cfg_parallel=False))._shared_cli_args(stage_mode="compile")
    assert "--cfg-parallel" in on
    assert "--cfg-parallel" not in off


@pytest.mark.parametrize(
    "module_path, cls_name, model_id",
    [
        # Flux is intentionally excluded: cfg-parallel is rejected for it at the
        # CLI, and its orchestrator no longer threads the flag.
        ("difflet.cli.orchestrators.ltx_2", "LTX2Orchestrator", "Lightricks/LTX-2"),
    ],
)
def test_inprocess_parallel_threads_cfg(module_path, cls_name, model_id):
    mod = importlib.import_module(module_path)
    orch_cls = getattr(mod, cls_name)
    args = argparse.Namespace(
        model_id=model_id, tp_degree=4, cp_degree=1, cp_mode="gather_kv",
        cfg_parallel=True,
    )
    parallel = orch_cls(args)._parallel()
    assert parallel.cfg_parallel_enabled is True
    # CFG parallel doubles the world size on top of TP.
    assert parallel.world_size == parallel.tp_degree * 2
