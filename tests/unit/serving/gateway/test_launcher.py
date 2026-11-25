from __future__ import annotations

import os
import signal
import socket
import subprocess
from pathlib import Path

import pytest

from difflet.bench import host
from difflet.serving.gateway import launcher
from difflet.serving.gateway.fleet import FleetConfig, FleetEntry
from difflet.serving.gateway.launcher import LauncherError, backend_argv, backend_env, down, up

FLUX = "black-forest-labs/FLUX.1-dev"
HV = "hunyuanvideo-community/HunyuanVideo"


def _entry(model_id=FLUX, device=0, port=8100, **kw):
    args = ("--tp-degree", "4", "--height", "1024", "--width", "1024")
    return FleetEntry(model_id=model_id, device=device, port=port,
                      cache_dir=Path(f"/c/{device}"), serve_args=args, **kw)


def _config(tmp_path, *entries):
    return FleetConfig(host="0.0.0.0", port=8000, runs_dir=tmp_path / "runs", models=entries)


def _closed_port():
    """A bound, not-listening socket: connecting to it is refused until ``listen()``."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    return sock, sock.getsockname()[1]


def _dead_pid():
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


@pytest.fixture
def fake_host(monkeypatch):
    devices = {d: host.Device(index=d, cores=tuple(range(4 * d, 4 * d + 4)), numa_node=d // 8,
                              cpu_affinity="") for d in range(16)}
    holders: dict[int, list[dict]] = {}
    monkeypatch.setattr(host, "topology", lambda: {"instance_type": "x", "devices": devices})
    monkeypatch.setattr(host, "numa_cpus", lambda: {0: list(range(48, 96)), 1: list(range(96))})
    monkeypatch.setattr(host, "neuronx_cc_dir", lambda path: None)
    monkeypatch.setattr(host, "device_holders",
                        lambda ds: {d: holders.get(d, []) for d in ds})
    monkeypatch.setattr(launcher, "POLL_S", 0.01)
    return holders


class _Popen:
    """Records each launch; ``on_poll`` runs on the first ``poll()``, ``exit_code`` ends it."""

    calls: list = []
    on_poll = None
    exit_code = None
    log_lines: list[str] = []

    def __init__(self, argv, **kwargs):
        self.argv, self.kwargs = argv, kwargs
        self.pid = 40000 + len(_Popen.calls)
        self.returncode = None
        self._polled = False
        for line in _Popen.log_lines:
            kwargs["stdout"].write(f"{line}\n".encode())
        _Popen.calls.append(self)

    def poll(self):
        if not self._polled and _Popen.on_poll:
            _Popen.on_poll(self)
        self._polled = True
        if _Popen.exit_code is not None:
            self.returncode = _Popen.exit_code
        return self.returncode


@pytest.fixture
def popen(monkeypatch):
    _Popen.calls, _Popen.on_poll, _Popen.exit_code, _Popen.log_lines = [], None, None, []
    monkeypatch.setattr(launcher.subprocess, "Popen", _Popen)
    return _Popen


def test_backend_env_device_2():
    env = backend_env(_entry(device=2), {"DIFFLET_API_KEY": "secret", "HOME": "/h"},
                      visible_cores="8-11", cc_dir=None)
    assert env["NEURON_RT_VISIBLE_CORES"] == "8-11"
    assert env["DIFFLET_API_KEY"] == ""
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["DIFFLET_BACKEND"] == "neuron"
    assert env["NEURON_NUMA_PIN"] == "0"
    assert env["HOME"] == "/h"


def test_backend_env_prepends_cc_dir():
    env = backend_env(_entry(), {"PATH": "/usr/bin"}, visible_cores="0-3", cc_dir="/x/bin")
    assert env["PATH"] == "/x/bin:/usr/bin"


def test_backend_env_merges_entry_env_under_the_fixed_variables():
    entry = _entry(env={"HF_HUB_CACHE": "/hub", "HOME": "/entry", "DIFFLET_API_KEY": "leak",
                        "NEURON_RT_VISIBLE_CORES": "0-63"})
    env = backend_env(entry, {"HOME": "/h", "HF_HUB_CACHE": "/root-hub"}, visible_cores="0-3",
                      cc_dir=None)
    assert env["HF_HUB_CACHE"] == "/hub"
    assert env["HOME"] == "/entry"
    assert env["DIFFLET_API_KEY"] == ""
    assert env["NEURON_RT_VISIBLE_CORES"] == "0-3"


def test_backend_argv():
    assert backend_argv(_entry(), python="/venv/bin/python", cpus="48-95,144-191") == [
        "taskset", "-c", "48-95,144-191", "/venv/bin/python", "-m", "difflet.cli.main", "serve",
        "--model-id", FLUX, "--host", "127.0.0.1", "--port", "8100", "--cache-dir", "/c/0",
        "--tp-degree", "4", "--height", "1024", "--width", "1024"]


def test_read_pid_requires_a_live_process_in_the_run_dir(tmp_path, monkeypatch):
    pid_file = tmp_path / "serve.pid"
    assert launcher.read_pid(pid_file) is None
    pid_file.write_text(f"{os.getpid()}\n")
    monkeypatch.chdir(tmp_path)
    assert launcher.read_pid(pid_file) == os.getpid()
    monkeypatch.chdir("/")  # a reused pid: alive, but not the backend started there
    assert launcher.read_pid(pid_file) is None
    pid_file.write_text(f"{_dead_pid()}\n")
    assert launcher.read_pid(pid_file) is None
    pid_file.write_text("garbage")
    assert launcher.read_pid(pid_file) is None


def test_up_refuses_busy_device(tmp_path, fake_host, popen):
    fake_host[0] = [{"pid": 999999, "comm": "python", "uid": 1, "NEURON_RT_VISIBLE_CORES": "0-3"}]
    with pytest.raises(LauncherError, match="999999"):
        up(_config(tmp_path, _entry()))
    assert popen.calls == []


def test_up_ignores_fleet_own_pids(tmp_path, fake_host, popen, monkeypatch):
    sock, port = _closed_port()
    other, mine = _entry(HV, device=1, port=8101), _entry(port=port)
    config = _config(tmp_path, mine, other)
    config.run_dir(other).mkdir(parents=True)
    config.pid_file(other).write_text(str(os.getpid()))
    monkeypatch.chdir(config.run_dir(other))  # the other entry's backend is this process
    fake_host[0] = [{"pid": os.getpid(), "comm": "python", "uid": 1,
                     "NEURON_RT_VISIBLE_CORES": "4-7"}]
    popen.on_poll = lambda proc: sock.listen()
    up(config)
    assert [p.argv[p.argv.index("--model-id") + 1] for p in popen.calls] == [FLUX]
    sock.close()


def test_up_skips_running_entry(tmp_path, fake_host, popen, monkeypatch):
    config = _config(tmp_path, _entry())
    config.run_dir(config.models[0]).mkdir(parents=True)
    config.pid_file(config.models[0]).write_text(str(os.getpid()))
    monkeypatch.chdir(config.run_dir(config.models[0]))
    up(config)
    assert popen.calls == []


def test_up_refuses_port_in_use(tmp_path, fake_host, popen):
    sock, port = _closed_port()
    sock.listen()
    with pytest.raises(LauncherError, match=f"port {port}"):
        up(_config(tmp_path, _entry(port=port)))
    assert popen.calls == []
    sock.close()


def test_up_waits_for_port_then_next(tmp_path, fake_host, popen):
    (sock_a, port_a), (sock_b, port_b) = _closed_port(), _closed_port()
    sockets = {port_a: sock_a, port_b: sock_b}
    listened = []  # (port, backends launched when it started listening)

    def listen(proc):
        port = int(proc.argv[proc.argv.index("--port") + 1])
        listened.append((port, len(popen.calls)))
        sockets[port].listen()

    popen.on_poll = listen
    config = _config(tmp_path, _entry(port=port_a), _entry(HV, device=9, port=port_b))
    up(config)
    assert listened == [(port_a, 1), (port_b, 2)]
    first, second = popen.calls
    assert first.argv[:3] == ["taskset", "-c", "48-95"]  # device 0: NUMA node 0's CPUs
    assert second.argv[:3] == ["taskset", "-c", "0-95"]  # device 9: node 1
    assert first.kwargs["env"]["NEURON_RT_VISIBLE_CORES"] == "0-3"
    assert second.kwargs["env"]["NEURON_RT_VISIBLE_CORES"] == "36-39"
    assert first.kwargs["cwd"] == config.run_dir(config.models[0])
    assert first.kwargs["start_new_session"] is True
    assert config.pid_file(config.models[0]).read_text().strip() == str(first.pid)
    for sock in sockets.values():
        sock.close()


def test_up_reports_log_tail_when_backend_exits(tmp_path, fake_host, popen):
    sock, port = _closed_port()
    popen.exit_code = 1
    popen.log_lines = [f"line {i}" for i in range(50)]
    with pytest.raises(LauncherError) as exc:
        up(_config(tmp_path, _entry(port=port)))
    message = str(exc.value)
    assert "status 1" in message and "line 49" in message and "line 10" in message
    assert "line 9\n" not in message
    sock.close()


def test_down_terms_then_kills(tmp_path, monkeypatch):
    config = _config(tmp_path, _entry())
    pid_file = config.pid_file(config.models[0])
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("4242")
    monkeypatch.setattr(launcher, "read_pid", lambda path: 4242 if path == pid_file else None)
    sent = []
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    down(config, grace_s=0.05)
    assert sent[0] == (4242, signal.SIGTERM)
    assert sent[-1] == (4242, signal.SIGKILL)
    assert not pid_file.exists()
