"""A stage launched by run_stage dies with the CLI process that launched it.

Otherwise a killed `difflet compile` leaves its stage compiling in the background, holding
NeuronCores and racing the next run over the same cache directory.
"""

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]  # the difflet the subprocess imports

_PARENT = textwrap.dedent("""
    import sys
    from difflet.cli import runner

    pidfile = sys.argv[1]
    child = ("import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); "
             "time.sleep(120)")
    runner.build_stage_command = lambda orchestrator, stage, cli_args, nproc=None: [
        sys.executable, "-c", child, pidfile]
    runner._is_whole_device = lambda num_cores: True  # never query the devices
    runner._prepare_mpmd_env = lambda env, num_cores, strict_environment=False: None
    runner.run_stage("unit/model", "generate", 1, None, [])
""")


def _gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{pid}/stat") as f:
                if f.read().split(") ")[-1].split()[0] in ("Z", "X"):
                    return True
        except FileNotFoundError:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
@pytest.mark.parametrize("backend", ["cpu", "neuron"])  # single stage process / torchrun launch
def test_a_stage_dies_with_the_cli_that_launched_it(tmp_path, backend):
    pidfile = tmp_path / "stage.pid"
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT, str(pidfile)],
        env={**os.environ, "PYTHONPATH": str(REPO), "DIFFLET_BACKEND": backend,
             "BASE_COMPILE_WORK_DIR": str(tmp_path / "work")},
    )
    deadline = time.monotonic() + 60
    while not (pidfile.exists() and pidfile.read_text()):
        assert time.monotonic() < deadline and parent.poll() is None, "stage never started"
        time.sleep(0.05)
    child = int(pidfile.read_text())
    try:
        os.kill(parent.pid, signal.SIGKILL)  # the CLI is killed (kill -9, OOM killer)
        parent.wait(timeout=10)
        assert _gone(child, timeout=10), "the stage outlived the CLI that launched it"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)
