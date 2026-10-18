import argparse
import os
import sys
import textwrap
from pathlib import Path

import pytest

import difflet
from difflet.cli.dp.claims import summarize
from difflet.cli.dp.requests_io import RequestSpec
import subprocess
import time
from difflet.cli.dp.router import (
    replica_core_ranges,
    run_router,
    worker_cli_args,
    worker_env,
)


def test_replica_core_ranges():
    assert replica_core_ranges(4, 4) == ["0-3", "4-7", "8-11", "12-15"]
    assert replica_core_ranges(2, 1) == ["0", "1"]


def test_worker_env_overrides_parent_values():
    base = {"NEURON_RT_NUM_CORES": "16", "PATH": "/bin"}
    env = worker_env(base, "4-7", 4)
    assert env["NEURON_RT_VISIBLE_CORES"] == "4-7"
    assert env["NEURON_RT_NUM_CORES"] == "4"
    assert env["PATH"] == "/bin"


def _args(**kw):
    defaults = dict(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers", tp_degree=4, cp_degree=None,
        cp_mode="gather_kv", cfg_parallel=False, sp_enabled=False, height=None,
        width=None, num_frames=None, steps=None, guidance_scale=None, seed=42,
        cache_dir=None, work_dir=None, keep_work_dir=False, dp=2,
        dp_schedule="round_robin", requests=None, revision=None, force=False,
        teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
        teacache_calibration=None, prompt=None, output=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_worker_cli_args_never_leak_router_flags():
    argv = worker_cli_args(_args(cp_degree=2, cfg_parallel=True, height=480))
    for forbidden in ("--dp", "--mode", "--prompt", "--output", "--requests"):
        assert forbidden not in argv
    text = " ".join(argv)
    assert "--model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers" in text
    assert "--cp-degree 2" in text and "--cfg-parallel" in text and "--height 480" in text


def test_worker_cli_args_forward_sp_and_keep_work_dir():
    argv = worker_cli_args(_args(sp_enabled=True, keep_work_dir=True))
    assert "--sp" in argv and "--keep-work-dir" in argv


def test_worker_cli_args_forward_torch_compile():
    # A CUDA replica compiles its DiT blocks exactly as a one-process run would.
    assert "--torch-compile" in worker_cli_args(_args(torch_compile=True))
    assert "--torch-compile" not in worker_cli_args(_args(torch_compile=False))
    assert "--torch-compile" not in worker_cli_args(_args())  # an older namespace


def test_check_hbm_skips_when_weights_missing(monkeypatch, capsys):
    from difflet.cli.dp import router as router_mod

    def raise_oserror(*a, **k):
        raise OSError("no local weights")

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", raise_oserror)
    router_mod._check_hbm(_args())
    assert "skipping HBM fit check" in capsys.readouterr().out


def test_check_hbm_runs_against_local_weights(monkeypatch, tmp_path):
    import json
    import struct

    from difflet.cli.dp import router as router_mod

    header = {"w": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}}
    blob = json.dumps(header).encode("utf-8")
    st = tmp_path / "transformer" / "m.safetensors"
    st.parent.mkdir(parents=True)
    st.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 32)
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path", lambda *a, **k: str(tmp_path)
    )
    router_mod._check_hbm(_args())  # tiny weights fit; no raise


# A stub "worker" used instead of difflet.cli.main: claims and completes requests.
_STUB = textwrap.dedent("""
    import argparse, sys
    from difflet.cli.dp import claims
    p = argparse.ArgumentParser()
    p.add_argument("command")
    p.add_argument("--requests-dir", required=True)
    p.add_argument("--worker-index", type=int, required=True)
    p.add_argument("--dp-schedule", required=True)
    p.add_argument("--fail-index", type=int, default=None)
    p.add_argument("--crash-worker", type=int, default=None)
    p.add_argument("--env-dump", default=None)
    args, _ = p.parse_known_args()
    if args.env_dump:
        import json, os
        keys = [k for k in os.environ if k.startswith(("NEURON_RT_", "CUDA_VISIBLE"))]
        with open(f"{args.env_dump}/worker_{args.worker_index}.json", "w") as fh:
            json.dump({k: os.environ[k] for k in keys}, fh)
    if args.crash_worker == args.worker_index:
        sys.exit(3)
    while (req := claims.claim_next(args.requests_dir, args.worker_index,
                                    args.dp_schedule)) is not None:
        if args.fail_index == req.index:
            claims.mark_failed(args.requests_dir, req.index, "stub failure")
        else:
            claims.mark_done(args.requests_dir, req.index)
""")


@pytest.fixture(autouse=True)
def _worktree_on_pythonpath(monkeypatch):
    """Stub workers run `python stub.py` (script-dir sys.path[0]), so the
    worktree's difflet must come from PYTHONPATH. Real workers use
    `python -m difflet.cli.main` from the caller's cwd and don't need this."""
    repo_root = str(Path(difflet.__file__).parents[1])
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH", repo_root + (os.pathsep + existing if existing else "")
    )


def _stub_prefix(tmp_path):
    stub = tmp_path / "stub_worker.py"
    stub.write_text(_STUB, encoding="utf-8")
    return [sys.executable, str(stub)]


def _requests(n):
    return [RequestSpec(index=i, prompt=f"p{i}", output=f"o{i}.png") for i in range(n)]


def test_run_router_all_success(tmp_path):
    args = _args(work_dir=str(tmp_path / "work"))
    rc = run_router(args, _requests(5), replica_cores=4,
                    worker_argv_prefix=_stub_prefix(tmp_path))
    assert rc == 0
    s = summarize(tmp_path / "work" / "requests")
    assert s.done == [0, 1, 2, 3, 4] and not s.failed and not s.unfinished


def test_run_router_reports_failed_request(tmp_path):
    args = _args(work_dir=str(tmp_path / "work"), dp_schedule="least_loaded")
    rc = run_router(args, _requests(4), replica_cores=4,
                    worker_argv_prefix=_stub_prefix(tmp_path) + ["--fail-index", "2"])
    assert rc != 0
    s = summarize(tmp_path / "work" / "requests")
    assert 2 in s.failed and sorted(s.done) == [0, 1, 3]


def test_run_router_crashed_worker_round_robin(tmp_path):
    args = _args(work_dir=str(tmp_path / "work"), dp=2)
    rc = run_router(args, _requests(4), replica_cores=4,
                    worker_argv_prefix=_stub_prefix(tmp_path) + ["--crash-worker", "1"])
    assert rc != 0
    s = summarize(tmp_path / "work" / "requests")
    assert sorted(s.done) == [0, 2]            # worker 0's assignments
    assert set(s.failed) == {1, 3}             # dead worker's assignments marked failed


# ------------------------------------------------- per-worker bootstrap root

def test_worker_env_gives_each_replica_its_own_root_comm_id():
    # ltx_2/dp2tp2 on device (2026-09-07): both workers bootstrapped their
    # 2-rank world through the same localhost:62182 root and hung for 4 h.
    base = {"NEURON_RT_ROOT_COMM_ID": "localhost:62182"}
    envs = [worker_env(base, r, 2) for r in replica_core_ranges(2, 2)]
    ids = [e["NEURON_RT_ROOT_COMM_ID"] for e in envs]
    assert len(set(ids)) == 2, ids
    assert "localhost:62182" not in ids  # the inherited, shared value is replaced
    for value in ids:
        host, port = value.rsplit(":", 1)
        assert host == "localhost" and 1024 <= int(port) <= 65535


def test_worker_env_honours_an_explicit_root_comm_id():
    env = worker_env({}, "0-1", 2, root_comm_id="localhost:50000")
    assert env["NEURON_RT_ROOT_COMM_ID"] == "localhost:50000"


# ------------------------------------------------- workers die with the router

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is Linux-only")
def test_workers_are_killed_when_the_router_dies(tmp_path):
    # A stand-in router spawns a long-lived worker exactly the way run_router
    # does (preexec_fn=_die_with_parent), reports the worker pid, then blocks.
    # SIGKILL the router: the worker must be gone within a couple of seconds.
    pidfile = tmp_path / "worker.pid"
    code = (
        "import subprocess, sys, time\n"
        "from difflet.cli.dp.router import _die_with_parent\n"
        "p = subprocess.Popen(['sleep', '300'], preexec_fn=_die_with_parent)\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(300)\n"
    )
    router = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 20
        while not pidfile.exists() or not pidfile.read_text().strip():
            assert time.monotonic() < deadline, "stand-in router never spawned its worker"
            time.sleep(0.05)
        worker_pid = int(pidfile.read_text())
        assert _alive(worker_pid)
    finally:
        router.kill()
        router.wait()
    deadline = time.monotonic() + 5
    while _alive(worker_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(worker_pid), "worker outlived the killed router"


def test_router_workers_keep_their_sigkill_death_signal(monkeypatch):
    """The DP workers' preexec stays SIGKILL (torchrun's child gets SIGTERM instead)."""
    import ctypes
    import signal

    from difflet.cli.dp.router import _die_with_parent

    prctl_calls = []

    class _Libc:
        def prctl(self, *args):
            prctl_calls.append(args)
            return 0

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda name, use_errno=False: _Libc())
    monkeypatch.setattr(os, "getppid", lambda: 4242)
    _die_with_parent()
    assert prctl_calls == [(1, int(signal.SIGKILL), 0, 0, 0)]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Zombies answer kill(0); treat a zombie as dead.
    try:
        with open(f"/proc/{pid}/status") as fh:
            return "State:\tZ" not in fh.read()
    except FileNotFoundError:
        return False


# ------------------------------------------------- CUDA replicas (F2 task 4)


def test_cuda_worker_env_pins_the_replica_gpus_and_drops_neuron_variables():
    base = {
        "NEURON_RT_NUM_CORES": "16",
        "NEURON_RT_VISIBLE_CORES": "0-15",
        "NEURON_RT_ROOT_COMM_ID": "localhost:62182",
        "PATH": "/bin",
    }
    env = worker_env(base, "2-3", 2, backend="cuda")
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3" and env["PATH"] == "/bin"
    assert not [k for k in env if k.startswith("NEURON_RT_")]
    assert worker_env({}, "1", 1, backend="cuda")["CUDA_VISIBLE_DEVICES"] == "1"


def test_cuda_worker_env_indexes_into_an_inherited_device_list():
    # The parent's CUDA_VISIBLE_DEVICES is the GPU budget the replica ranges count from:
    # replica "1" of a parent restricted to GPUs 4,6 is physical GPU 6, not GPU 1.
    base = {"CUDA_VISIBLE_DEVICES": "4,6"}
    assert worker_env(base, "0", 1, backend="cuda")["CUDA_VISIBLE_DEVICES"] == "4"
    assert worker_env(base, "1", 1, backend="cuda")["CUDA_VISIBLE_DEVICES"] == "6"
    assert worker_env(base, "0-1", 2, backend="cuda")["CUDA_VISIBLE_DEVICES"] == "4,6"
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        worker_env(base, "2", 1, backend="cuda")


def test_trainium_worker_env_is_the_default_and_sets_no_cuda_devices():
    env = worker_env({"PATH": "/bin"}, "0-3", 4)
    assert "CUDA_VISIBLE_DEVICES" not in env and env["NEURON_RT_VISIBLE_CORES"] == "0-3"


def test_run_router_on_cuda_gives_each_worker_its_own_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "16")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dump = tmp_path / "env"
    dump.mkdir()
    args = _args(work_dir=str(tmp_path / "work"), dp=2, tp_degree=None)
    rc = run_router(
        args,
        _requests(4),
        replica_cores=1,
        worker_argv_prefix=_stub_prefix(tmp_path) + ["--env-dump", str(dump)],
    )
    assert rc == 0
    import json

    envs = [json.loads((dump / f"worker_{w}.json").read_text()) for w in (0, 1)]
    assert envs == [{"CUDA_VISIBLE_DEVICES": "0"}, {"CUDA_VISIBLE_DEVICES": "1"}]
    s = summarize(tmp_path / "work" / "requests")
    assert s.done == [0, 1, 2, 3] and not s.failed and not s.unfinished


def test_run_router_on_cuda_two_gpu_replicas(tmp_path, monkeypatch):
    # dp=2 x cfg=2: each replica is a torchrun world on its own pair of GPUs.
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dump = tmp_path / "env"
    dump.mkdir()
    args = _args(work_dir=str(tmp_path / "work"), dp=2, tp_degree=None, cfg_parallel=True)
    rc = run_router(
        args,
        _requests(2),
        replica_cores=2,
        worker_argv_prefix=_stub_prefix(tmp_path) + ["--env-dump", str(dump)],
    )
    assert rc == 0
    import json

    envs = [json.loads((dump / f"worker_{w}.json").read_text()) for w in (0, 1)]
    assert envs == [{"CUDA_VISIBLE_DEVICES": "0,1"}, {"CUDA_VISIBLE_DEVICES": "2,3"}]


def test_run_router_on_trainium_keeps_the_neuron_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dump = tmp_path / "env"
    dump.mkdir()
    args = _args(work_dir=str(tmp_path / "work"), dp=2)
    rc = run_router(
        args,
        _requests(2),
        replica_cores=4,
        worker_argv_prefix=_stub_prefix(tmp_path) + ["--env-dump", str(dump)],
    )
    assert rc == 0
    import json

    env0 = json.loads((dump / "worker_0.json").read_text())
    assert env0["NEURON_RT_VISIBLE_CORES"] == "0-3" and "CUDA_VISIBLE_DEVICES" not in env0
