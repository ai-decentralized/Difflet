from __future__ import annotations

from pathlib import Path

import pytest

from difflet.cli import gateway as gateway_cli
from difflet.cli.main import _build_parser, main
from difflet.serving.auth import ApiKeyAuthenticationMiddleware

EXAMPLE = str(Path(__file__).resolve().parents[3] / "examples" / "serving" / "fleet.yaml")


@pytest.mark.parametrize("action", ["serve", "up", "down", "status"])
def test_gateway_parser_actions(action):
    args = _build_parser().parse_args(["gateway", action, "--config", "f.yaml"])
    assert (args.command, args.gateway_action, args.config) == ("gateway", action, "f.yaml")
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["gateway", action])


@pytest.mark.parametrize("action", ["up", "down"])
def test_gateway_up_calls_launcher(monkeypatch, action):
    calls = []
    monkeypatch.setattr(gateway_cli, action, lambda config: calls.append(config))
    main(["gateway", action, "--config", EXAMPLE])
    (config,) = calls
    assert config.models[0].model_id == "black-forest-labs/FLUX.1-dev"


def test_gateway_serve_builds_app(monkeypatch):
    import uvicorn

    runs = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: runs.append((app, kw)))
    monkeypatch.delenv("DIFFLET_API_KEY", raising=False)
    main(["gateway", "serve", "--config", EXAMPLE, "--api-key", "k"])
    ((app, kw),) = runs
    assert (kw["host"], kw["port"], kw["workers"]) == ("0.0.0.0", 8000, 1)
    assert any(m.cls is ApiKeyAuthenticationMiddleware for m in app.user_middleware)
    main(["gateway", "serve", "--config", EXAMPLE])
    assert not runs[1][0].user_middleware  # no key anywhere: no auth


def test_gateway_status_prints_rows(monkeypatch, capsys):
    async def fake_status(config, client):
        return {e.model_id: "loading" for e in config.models}

    monkeypatch.setattr(gateway_cli, "fleet_status", fake_status)
    main(["gateway", "status", "--config", EXAMPLE])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["model", "device", "port", "status"]
    assert lines[1].split() == ["black-forest-labs/FLUX.1-dev", "0", "8100", "loading"]
    assert len(lines) == 7


def test_gateway_config_error_exits_1(tmp_path, capsys):
    bad = tmp_path / "fleet.yaml"
    bad.write_text("models: []\nruns_dir: /x\n")
    with pytest.raises(SystemExit) as exc:
        main(["gateway", "up", "--config", str(bad)])
    assert exc.value.code == 1
    assert capsys.readouterr().err.startswith("Error: models")
