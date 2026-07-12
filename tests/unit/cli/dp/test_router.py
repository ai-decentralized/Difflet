import argparse
import os
import sys
import textwrap
from pathlib import Path

import pytest

import difflet
from difflet.cli.dp.claims import summarize
from difflet.cli.dp.requests_io import RequestSpec
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
    args, _ = p.parse_known_args()
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
