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


def test_compile_accepts_adaptive_teacache_flags(monkeypatch, tmp_path):
    """The probe NEFF is part of the artifact, so `difflet compile` takes the
    calibrated-adaptive flags (flux: a separate probe identity; qwen_image /
    hunyuan_video: an additive probe component) and routes them to the
    orchestrator exactly like generate does."""
    cli = _cli()
    seen = {}

    class FakeOrch:
        def compile(self):
            return None

    def fake_orchestrator(args):
        seen.update(vars(args))
        return FakeOrch()

    monkeypatch.setattr(cli, "_get_orchestrator", fake_orchestrator)
    calib = tmp_path / "calib.json"
    calib.write_text("{}")
    cli.main(["compile", "--model-id", "black-forest-labs/FLUX.1-dev",
              "--teacache-speedup", "1.5", "--teacache-calibration", str(calib)])
    assert seen["teacache_speedup"] == 1.5
    assert seen["teacache_calibration"] == str(calib)


def test_compile_teacache_speedup_requires_calibration(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["compile", "--model-id", "black-forest-labs/FLUX.1-dev",
                  "--teacache-speedup", "1.5"])
    assert exc.value.code == 1
    assert "--teacache-calibration" in capsys.readouterr().err
