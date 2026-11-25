from __future__ import annotations

from pathlib import Path

import pytest

from starlette.middleware.cors import CORSMiddleware

from difflet.cli import gateway as gateway_cli
from difflet.cli.main import _build_parser, main
from difflet.serving.auth import ApiKeyAuthenticationMiddleware

EXAMPLE = str(Path(__file__).resolve().parents[3] / "examples" / "serving" / "fleet.yaml")


@pytest.mark.parametrize("action", ["serve", "up", "down", "status"])
def test_gateway_parser_actions(action, capsys):
    args = _build_parser().parse_args(["gateway", action, "--config", "f.yaml"])
    assert (args.command, args.gateway_action, args.config) == ("gateway", action, "f.yaml")
    with pytest.raises(SystemExit) as exc:
        main(["gateway", action])
    assert exc.value.code == 1 and "--config" in capsys.readouterr().err


def test_gateway_page_writes_the_static_page(tmp_path, capsys):
    out = tmp_path / "site" / "index.html"
    out.parent.mkdir()
    main(["gateway", "page", "--api-base", "https://api.example", "--out", str(out)])
    page = out.read_text()
    assert page.startswith("<!doctype html>")
    assert 'window.DIFFLET_API = "https://api.example";' in page and "<table" not in page
    assert str(out) in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--out", "x.html"], ["--api-base", "https://api.example"],
                                  ["--api-base", "api.example", "--out", "x.html"]])
def test_gateway_page_needs_an_api_base_url_and_an_out_path(tmp_path, capsys, argv):
    with pytest.raises(SystemExit) as exc:
        main(["gateway", "page", *argv])
    assert exc.value.code == 1 and capsys.readouterr().err.startswith("Error: ")


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
    assert (kw["host"], kw["port"], kw["workers"]) == ("127.0.0.1", 8000, 1)
    assert [m.cls for m in app.user_middleware] == [CORSMiddleware, ApiKeyAuthenticationMiddleware]
    main(["gateway", "serve", "--config", EXAMPLE])
    assert [m.cls for m in runs[1][0].user_middleware] == [CORSMiddleware]  # no key: no auth


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
