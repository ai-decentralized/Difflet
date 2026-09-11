from __future__ import annotations
import importlib
import pytest


def _cli() -> object:
    return importlib.import_module("difflet.cli.main")


def test_unknown_model_id_exits(capsys):
    cli = _cli()
    with pytest.raises(SystemExit) as exc:
        cli.main(["compile", "--model-id", "badmodel"])
    assert exc.value.code == 1
    assert "Unknown model-id 'badmodel'" in capsys.readouterr().err


def test_teacache_cadence_and_online_delta_mutually_exclusive(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "generate", "--model-id", "black-forest-labs/FLUX.1-dev",
            "--prompt", "x", "--output", "out.png",
            "--teacache-cadence", "2", "--teacache-online-delta", "0.6",
        ])
    assert exc.value.code == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_teacache_speedup_requires_calibration(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "generate", "--model-id", "black-forest-labs/FLUX.1-dev",
            "--prompt", "x", "--output", "out.png",
            "--teacache-speedup", "1.5",
        ])
    assert exc.value.code == 1
    assert "--teacache-calibration" in capsys.readouterr().err


def test_valid_generate_routes_to_orchestrator(monkeypatch):
    cli = _cli()
    calls = []
    class FakeOrch:
        def generate(self): calls.append("generate")
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: FakeOrch())
    cli.main(["generate", "--model-id", "black-forest-labs/FLUX.1-dev",
              "--prompt", "x", "--output", "o.png"])
    assert calls == ["generate"]


def test_taef1_requires_model_path(capsys):
    cli = _cli()
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "generate", "--model-id", "black-forest-labs/FLUX.1-dev",
            "--prompt", "x", "--output", "o.png", "--taef1",
        ])
    assert exc.value.code == 1
    assert "--taef1-path" in capsys.readouterr().err


def test_taef1_path_implies_taef1(monkeypatch):
    cli = _cli()
    captured = {}

    class FakeOrch:
        def generate(self):
            return None

    def fake_orchestrator(args):
        captured["taef1"] = args.taef1
        captured["taef1_path"] = args.taef1_path
        return FakeOrch()

    monkeypatch.setattr(cli, "_get_orchestrator", fake_orchestrator)
    cli.main([
        "generate", "--model-id", "black-forest-labs/FLUX.1-dev",
        "--prompt", "x", "--output", "o.png",
        "--taef1-path", "madebyollin/taef1",
    ])
    assert captured == {
        "taef1": True,
        "taef1_path": "madebyollin/taef1",
    }


def test_generate_refuses_the_tpu_backend_before_spawning_stages(monkeypatch, capsys):
    """v5e: `difflet generate` for a TPU-ported model spawned difflet.cli.stage,
    which imports the Neuron toolchain and died with ModuleNotFoundError."""
    cli = _cli()
    monkeypatch.setenv("DIFFLET_BACKEND", "tpu")
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: pytest.fail("orchestrator built"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["generate", "--model-id", "Qwen/Qwen-Image", "--prompt", "x", "--output", "o.png"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not available on the 'tpu' backend" in err
    assert "difflet serve" in err
