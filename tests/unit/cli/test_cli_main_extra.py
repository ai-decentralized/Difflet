from __future__ import annotations

import argparse

import pytest

import importlib

cli_main = importlib.import_module("difflet.cli.main")


def _ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="black-forest-labs/FLUX.1-dev",
        command="generate",
        tp_degree=None, cp_degree=1, cp_mode="gather_kv", cfg_parallel=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


_EXPECTED = [
    ("black-forest-labs/FLUX.1-dev",
     "difflet.cli.orchestrators.flux", "FluxOrchestrator"),
    ("Lightricks/LTX-2",
     "difflet.cli.orchestrators.ltx_2", "LTX2Orchestrator"),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers",
     "difflet.cli.orchestrators.wan", "WanOrchestrator"),
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers",
     "difflet.cli.orchestrators.wan", "WanOrchestrator"),
    ("hunyuanvideo-community/HunyuanVideo",
     "difflet.cli.orchestrators.hunyuan_video", "HunyuanVideoOrchestrator"),
    ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
     "difflet.cli.orchestrators.hunyuan_video_15", "HunyuanVideo15Orchestrator"),
    ("Qwen/Qwen-Image",
     "difflet.cli.orchestrators.qwen_image", "QwenImageOrchestrator"),
    ("MiniMaxAI/MiniMax-H3",
     "difflet.cli.orchestrators.minimax_h3", "MiniMaxH3Orchestrator"),
]


@pytest.mark.parametrize("model_id,module,clsname", _EXPECTED)
def test_get_orchestrator_returns_correct_type(model_id, module, clsname):
    import importlib
    expected_cls = getattr(importlib.import_module(module), clsname)
    orch = cli_main._get_orchestrator(_ns(model_id=model_id))
    assert isinstance(orch, expected_cls)
    assert orch.args.model_id == model_id


def test_get_orchestrator_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        cli_main._get_orchestrator(_ns(model_id="nope"))


def test_main_compile_invokes_compile_command(monkeypatch):
    calls = []

    class FakeOrch:
        def compile(self):
            calls.append("compile")

    monkeypatch.setattr(cli_main, "_get_orchestrator", lambda args: FakeOrch())
    cli_main.main([
        "compile", "--model-id", "black-forest-labs/FLUX.1-dev",
        "--tp-degree", "4",
    ])
    assert calls == ["compile"]


def test_main_download_skips_cfg_and_teacache_validation(monkeypatch):
    # download is not in the cfg/teacache validated commands; ensure it routes.
    calls = []

    class FakeOrch:
        def download(self):
            calls.append("download")

    monkeypatch.setattr(cli_main, "_get_orchestrator", lambda args: FakeOrch())
    cli_main.main(["download", "--model-id", "Qwen/Qwen-Image"])
    assert calls == ["download"]


def test_cfg_parallel_rejected_with_cp_degree(monkeypatch, capsys):
    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")  # AoT: no composing mesh (CI auto-detects it too)
    with pytest.raises(SystemExit) as exc:
        cli_main.main([
            "generate", "--model-id", "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--prompt", "x", "--output", "o.mp4",
            "--cfg-parallel", "--cp-degree", "2",
        ])
    assert exc.value.code == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_cfg_parallel_rejected_for_distilled_model(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main.main([
            "generate", "--model-id", "Qwen/Qwen-Image",
            "--prompt", "x", "--output", "o.png", "--cfg-parallel",
        ])
    assert exc.value.code == 1
    assert "guidance-distilled" in capsys.readouterr().err


def test_cfg_parallel_with_cp_degree_passes_validation_on_neuron():
    import argparse

    cli_main._validate_cfg_parallel(argparse.Namespace(cfg_parallel=True, cp_degree=2, backend="neuron",
                                                       model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers"))


@pytest.mark.parametrize(
    "extra",
    [[], ["--cfg-parallel"], ["--dp", "2", "--tp-degree", "2"], ["--mode", "latency"]],
)
def test_replica_cores_match_stage_layout(extra):
    """The DP router's replica size must equal what the hv15 transformer stage launches.

    Covers the Trainium path (no platform default tp), where the registry default layout
    (TP2 x CFG2) applies when no tp/cfg/cp is given; the CUDA semantics are pinned in
    test_main_dp.py.
    """
    from difflet.cli.orchestrators.hunyuan_video_15 import _layout

    model = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    args = cli_main._build_parser().parse_args(
        ["generate", "--model-id", model, "--prompt", "p", "--output", "o.mp4", *extra])
    tp, cfg, cp = _layout(args)
    assert cli_main._replica_cores(args) == tp * (2 if cfg else 1) * cp
