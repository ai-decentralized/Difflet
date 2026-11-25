from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import httpx
import pytest

from difflet.serving.gateway.fleet import FleetConfig, FleetEntry
from difflet.serving.gateway.status import backend_status, fleet_status

FLUX = "black-forest-labs/FLUX.1-dev"
QWEN = "Qwen/Qwen-Image"


def _entry(model_id=FLUX, port=8100):
    return FleetEntry(model_id=model_id, device=port - 8100, port=port, cache_dir=Path("/c"),
                      serve_args=("--height", "1024", "--width", "1024"))


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _answer(status, model=FLUX):
    def handler(request):
        assert request.url.path == "/ready"
        if status is None:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(status, json={"status": "x", "model": model})
    return handler


@pytest.fixture
def alive_pid_file(tmp_path, monkeypatch):
    pid_file = tmp_path / "serve.pid"
    pid_file.write_text(str(os.getpid()))
    monkeypatch.chdir(tmp_path)  # read_pid wants the process running in the pid file's dir
    return pid_file


@pytest.fixture
def dead_pid_file(tmp_path):
    proc = subprocess.Popen(["true"])
    proc.wait()
    pid_file = tmp_path / "dead.pid"
    pid_file.write_text(str(proc.pid))
    return pid_file


def _status(pid_file, handler):
    async def run():
        async with _client(handler) as client:
            return await backend_status(_entry(), pid_file, client)
    return asyncio.run(run())


@pytest.mark.parametrize("answer,expected", [
    (200, "ready"), (None, "loading"), (503, "recovering"), (500, "down")])
def test_status_matrix(alive_pid_file, dead_pid_file, answer, expected):
    assert _status(alive_pid_file, _answer(answer)) == expected
    assert _status(dead_pid_file, _answer(answer)) == "down"
    assert _status(alive_pid_file.parent / "missing.pid", _answer(answer)) == "down"


def test_ready_for_another_model_is_down(alive_pid_file):
    assert _status(alive_pid_file, _answer(200, model=QWEN)) == "down"


def test_fleet_status_probes_every_backend(tmp_path, monkeypatch):
    config = FleetConfig(host="0.0.0.0", port=8000, runs_dir=tmp_path,
                         models=(_entry(FLUX, 8100), _entry(QWEN, 8101)))
    for entry in config.models:
        config.run_dir(entry).mkdir()
        config.pid_file(entry).write_text(str(os.getpid()))
    monkeypatch.chdir(config.run_dir(config.models[1]))  # only Qwen's pid file is "ours"

    def handler(request):
        return httpx.Response(200, json={"model": QWEN if request.url.port == 8101 else FLUX})

    async def run():
        async with _client(handler) as client:
            return await fleet_status(config, client)

    assert asyncio.run(run()) == {FLUX: "down", QWEN: "ready"}
