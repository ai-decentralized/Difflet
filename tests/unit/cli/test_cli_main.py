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
