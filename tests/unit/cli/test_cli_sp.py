"""CLI surface for Megatron-style sequence parallelism (``--sp``).

Pure argparse / config-construction / validation tests (no Neuron runtime).
"""

from __future__ import annotations

import argparse
import importlib

import pytest

from difflet.cli.main import _build_parser


def _cli():
    return importlib.import_module("difflet.cli.main")


# --------------------------------------------------------------- flag parsing

def test_sp_flag_defaults_to_false():
    parser = _build_parser()
    args = parser.parse_args(["compile", "--model-id", "x"])
    assert args.sp_enabled is False


def test_sp_flag_sets_true():
    parser = _build_parser()
    args = parser.parse_args(["compile", "--model-id", "x", "--sp"])
    assert args.sp_enabled is True


# ----------------------------------------------------------------- validation

@pytest.mark.parametrize(
    "model_id",
    [
        "black-forest-labs/FLUX.1-dev",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "hunyuanvideo-community/HunyuanVideo",
        "Qwen/Qwen-Image",
    ],
)
def test_sp_allowed_for_supported_models(model_id, monkeypatch):
    cli = _cli()
    calls = []

    class FakeOrch:
        def compile(self):
            calls.append("compile")

    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: FakeOrch())
    cli.main(["compile", "--model-id", model_id, "--sp"])
    assert calls == ["compile"]


@pytest.mark.parametrize(
    "model_id",
    [
        "Lightricks/LTX-2",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    ],
)
def test_sp_rejected_for_unsupported_models(model_id, monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["compile", "--model-id", model_id, "--sp"])
    assert exc.value.code == 1
    assert "does not support --sp" in capsys.readouterr().err


def test_sp_with_cp_degree_rejected(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "compile", "--model-id", "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "--sp", "--cp-degree", "2",
        ])
    assert exc.value.code == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_sp_off_is_a_noop_for_validation(monkeypatch):
    cli = _cli()
    calls = []

    class FakeOrch:
        def compile(self):
            calls.append("compile")

    # No --sp: even an unsupported model must pass the SP guard untouched.
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: FakeOrch())
    cli.main(["compile", "--model-id", "Lightricks/LTX-2"])
    assert calls == ["compile"]


# --------------------------------------------------- orchestrator threading

def test_flux_parallel_threads_sp():
    from difflet.cli.orchestrators.flux import FluxOrchestrator

    args = argparse.Namespace(
        model_id="black-forest-labs/FLUX.1-dev",
        tp_degree=4, cp_degree=1, cp_mode="gather_kv", sp_enabled=True,
    )
    parallel = FluxOrchestrator(args)._parallel()
    assert parallel.sp_enabled is True
    # SP reuses the TP group: world_size is unchanged.
    assert parallel.world_size == parallel.tp_degree


def test_flux_parallel_sp_off_by_default():
    from difflet.cli.orchestrators.flux import FluxOrchestrator

    args = argparse.Namespace(
        model_id="black-forest-labs/FLUX.1-dev",
        tp_degree=4, cp_degree=1, cp_mode="gather_kv", sp_enabled=False,
    )
    assert FluxOrchestrator(args)._parallel().sp_enabled is False


def _staged_args(model_id, **over):
    base = dict(
        model_id=model_id, tp_degree=4, cp_degree=1, cp_mode="gather_kv",
        cfg_parallel=False, sp_enabled=True, height=None, width=None,
        num_frames=None, steps=None, guidance_scale=None, seed=42,
        prompt="p", output="o.png", cache_dir=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    "module_path, cls_name, model_id, stage",
    [
        ("difflet.cli.orchestrators.wan", "WanOrchestrator",
         "Wan-AI/Wan2.1-T2V-14B-Diffusers", "transformer"),
        ("difflet.cli.orchestrators.hunyuan_video", "HunyuanVideoOrchestrator",
         "hunyuanvideo-community/HunyuanVideo", "generate"),
        ("difflet.cli.orchestrators.qwen_image", "QwenImageOrchestrator",
         "Qwen/Qwen-Image", "generate"),
    ],
)
def test_staged_compiled_dir_marks_sp(module_path, cls_name, model_id, stage):
    mod = importlib.import_module(module_path)
    orch_cls = getattr(mod, cls_name)
    with_sp = orch_cls(_staged_args(model_id))._stage_compiled_dir(
        stage, _staged_args(model_id, sp_enabled=True)
    )
    without = orch_cls(_staged_args(model_id))._stage_compiled_dir(
        stage, _staged_args(model_id, sp_enabled=False)
    )
    assert "sp" in with_sp.name
    assert "sp" not in without.name
    assert with_sp != without


@pytest.mark.parametrize(
    "module_path, cls_name, model_id",
    [
        ("difflet.cli.orchestrators.wan", "WanOrchestrator",
         "Wan-AI/Wan2.1-T2V-14B-Diffusers"),
        ("difflet.cli.orchestrators.hunyuan_video", "HunyuanVideoOrchestrator",
         "hunyuanvideo-community/HunyuanVideo"),
        ("difflet.cli.orchestrators.qwen_image", "QwenImageOrchestrator",
         "Qwen/Qwen-Image"),
    ],
)
def test_staged_shared_cli_args_forwards_sp(module_path, cls_name, model_id):
    mod = importlib.import_module(module_path)
    orch_cls = getattr(mod, cls_name)
    on = orch_cls(_staged_args(model_id, sp_enabled=True))._shared_cli_args(stage_mode="compile")
    off = orch_cls(_staged_args(model_id, sp_enabled=False))._shared_cli_args(stage_mode="compile")
    assert "--sp" in on
    assert "--sp" not in off
