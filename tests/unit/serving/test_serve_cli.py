from __future__ import annotations

import argparse
import importlib

import pytest

from difflet.serving.cli.serve import options_from_args


def test_difflet_serve_routes_to_serving_command(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(args)

    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setattr("difflet.serving.cli.serve.run", fake_run)
    cli_main.main(["serve", "--model-id", "black-forest-labs/FLUX.1-dev", "--port", "9000"])

    assert calls
    assert calls[0].command == "serve"
    assert calls[0].port == 9000


def test_serve_help_hides_internal_worker_tuning(capsys):
    cli_main = importlib.import_module("difflet.cli.main")

    with pytest.raises(SystemExit):
        cli_main.main(["serve", "--help"])

    out = capsys.readouterr().out
    assert "--max-running-requests" not in out
    assert "--worker-cancel-timeout" not in out
    assert "--worker-restart-timeout" not in out
    assert "--max-queued-requests" not in out
    assert "--queue-timeout" not in out
    assert "--request-timeout" not in out
    assert "--artifact-store-timeout" not in out
    assert "--download-policy" not in out
    assert "--compile-policy" not in out
    assert "--artifact-store" not in out
    assert "--artifact-ttl-seconds" not in out
    assert "--cfg-parallel" not in out
    assert "--sp" not in out
    assert "--host-vae" not in out
    assert "--num-frames" not in out
    assert "Wan-AI/Wan2.2-T2V-A14B-Diffusers" not in out
    assert "hunyuanvideo-community/HunyuanVideo" not in out
    assert "Lightricks/LTX-2" not in out
    assert "FLUX.1-dev" in out
    assert "Qwen/Qwen-Image" in out


def test_serve_options_rejects_num_frames(capsys):
    args = argparse.Namespace(
        model_id="black-forest-labs/FLUX.1-dev",
        revision=None,
        host="0.0.0.0",
        port=8091,
        tp_degree=None,
        cp_degree=None,
        cp_mode=None,
        cfg_parallel=False,
        sp_enabled=False,
        height=None,
        width=None,
        num_frames=1,
        cache_dir=None,
        force=False,
    )

    with pytest.raises(SystemExit):
        options_from_args(args)

    assert "--num-frames" in capsys.readouterr().err
