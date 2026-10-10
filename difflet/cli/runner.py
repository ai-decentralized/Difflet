from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping

from difflet import envs
from difflet.common.neuron_cores import (
    resolve_available_neuron_core_ids,
    select_neuron_core_ids,
)

# Rank variables owned by torchrun. A copy inherited from the caller (run_stage
# called inside a process torchrun itself launched) must not reach the new launch.
_TORCHRUN_RANK_ENV: tuple[str, ...] = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
# torchrun's grace period for its ranks after it forwards a stop signal (torch.distributed
# LaunchConfig.shutdown_timeout), and the extra time an interrupted launch gives torchrun.
_TORCHRUN_SHUTDOWN_TIMEOUT_ENV = "TORCH_ELASTIC_SHUTDOWN_TIMEOUT"
_TORCHRUN_SHUTDOWN_DEFAULT_S = 30
_TORCHRUN_STOP_MARGIN_S = 30


def _is_whole_device(num_cores: int) -> bool:
    """True when num_cores already covers every visible core.

    Whole-device stages must NOT export ``NEURON_RT_NUM_CORES``: the default
    allocation already covers the visible set, and on some driver builds an
    explicit request naming the full count is rejected
    ("must request one core, or the whole device") even though the count equals
    the whole device — the default path allocates fine.
    """
    try:
        visible = resolve_available_neuron_core_ids(required_num_cores=num_cores)
        return num_cores >= len(visible)
    except Exception:
        return False


def build_stage_command(
    orchestrator: str,
    stage: str,
    cli_args: list[str],
    *,
    nproc: int | None = None,
) -> list[str]:
    """Command line of one stage launch.

    ``nproc=None`` is the single stage process every AoT backend uses (unchanged).
    An int runs ``nproc`` ranks of the same stage module under torchrun, one per
    NeuronCore, for a non-AoT torchrun-MPMD backend.
    """
    stage_module = [
        "-m",
        "difflet.cli.stage",
        "--orchestrator",
        orchestrator,
        "--stage",
        stage,
        *cli_args,
    ]
    if nproc is None:
        return [sys.executable, *stage_module]
    if nproc < 1:
        raise ValueError(f"nproc must be >= 1 for a torchrun launch, got {nproc}")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(nproc),
        *stage_module,
    ]


def _torchrun_mpmd_selected() -> bool:
    """True when DIFFLET_BACKEND explicitly names a non-AoT torchrun-MPMD backend.

    Opt-in only: auto-detection already resolves to ``neuron`` on a TorchNeuron
    host (difflet/backends/registry.py, _auto_detect_backend), so keying on the
    resolved backend would silently change every existing stage launch there. An
    unknown name raises ValueError here, before anything is spawned.
    """
    name = envs.DIFFLET_BACKEND
    if not name:
        return False
    from difflet.backends import get_backend

    capabilities = get_backend(name).capabilities
    return capabilities.supports_torchrun_mpmd and not capabilities.requires_aot


def _prepare_mpmd_env(env: dict[str, str], num_cores: int, *, strict_environment: bool) -> None:
    """Environment for ``num_cores`` torchrun ranks, one per NeuronCore.

    Each rank binds its own core from NEURON_RT_VISIBLE_CORES
    (difflet/backends/neuron/runtime.py, bind_core), so NEURON_RT_NUM_CORES and
    NEURON_RT_VIRTUAL_CORE_SIZE, which size one process's multi-core allocation,
    are never set here.
    """
    for name in _TORCHRUN_RANK_ENV:
        env.pop(name, None)
    if strict_environment:
        visible_core_ids = select_neuron_core_ids(required_num_cores=num_cores)
        env["NEURON_RT_VISIBLE_CORES"] = ",".join(str(core_id) for core_id in visible_core_ids)
        # A stale inherited count would contradict the per-rank binding.
        env.pop("NEURON_RT_NUM_CORES", None)


def _torchrun_stop_timeout(env: Mapping[str, str]) -> float:
    """How long an interrupted launch waits for torchrun before killing it.

    torchrun signals every rank, waits TORCH_ELASTIC_SHUTDOWN_TIMEOUT (30 s unless set; parsed
    with int() as torchrun does) and then SIGKILLs the ranks still running; the margin covers
    that and its own teardown.
    """
    try:
        shutdown_s = int(env.get(_TORCHRUN_SHUTDOWN_TIMEOUT_ENV, _TORCHRUN_SHUTDOWN_DEFAULT_S))
    except ValueError:
        shutdown_s = _TORCHRUN_SHUTDOWN_DEFAULT_S
    return float(max(shutdown_s, 0) + _TORCHRUN_STOP_MARGIN_S)


def _run_torchrun(cmd: list[str], env: dict[str, str]) -> None:
    """``subprocess.run(cmd, env=env, check=True)`` for a torchrun launch, except that torchrun
    is never killed before it has stopped its ranks.

    torchrun starts every rank in a session of its own, so a terminal's Ctrl-C reaches torchrun
    (in this process's group) but not the ranks, and only torchrun can stop them: it forwards
    the signal, waits, then SIGKILLs the ranks still running (one blocked in a native Neuron
    compile or collective cannot run Python's SIGINT handler). ``subprocess.run`` SIGKILLs
    torchrun 0.25 s after a KeyboardInterrupt, which would leave such ranks holding their
    NeuronCores. On KeyboardInterrupt this waits for torchrun instead (another Ctrl-C does not
    cut the wait short; torchrun receives that one too), kills it only after
    ``_torchrun_stop_timeout``, and re-raises. Any other exception first sends torchrun
    SIGTERM, which it handles like SIGINT. A non-zero exit raises CalledProcessError.
    """
    proc = subprocess.Popen(cmd, env=env)
    try:
        returncode = proc.wait()
    except BaseException as exc:
        if not isinstance(exc, KeyboardInterrupt):
            proc.terminate()
        _await_torchrun_exit(proc, _torchrun_stop_timeout(env))
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, cmd)


def _await_torchrun_exit(proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            proc.wait(timeout=max(deadline - time.monotonic(), 0.0))
            return
        except KeyboardInterrupt:
            continue  # Ctrl-C again: torchrun got it as well and is still stopping its ranks
        except subprocess.TimeoutExpired:
            break
    print(
        f"[difflet] torchrun (pid {proc.pid}) did not exit {timeout:.0f}s after being "
        "interrupted; killing it (its ranks may still hold NeuronCores)",
        file=sys.stderr,
        flush=True,
    )
    proc.kill()
    proc.wait()


def run_stage(
    orchestrator: str,
    stage: str,
    num_cores: int,
    virtual_core_size: int | None,
    cli_args: list[str],
    strict_environment: bool = False,
) -> None:
    """Spawn a stage subprocess with the correct Neuron env vars.

    Staged CLI calls preserve explicit shell overrides. Serving compilation uses
    ``strict_environment=True`` so the compiled topology matches its identity.
    When DIFFLET_BACKEND names a non-AoT torchrun-MPMD backend (neuron), the stage
    runs as ``num_cores`` torchrun ranks and ``virtual_core_size`` does not apply; an
    interrupt then waits for torchrun to stop its ranks (``_run_torchrun``).
    """
    nproc = num_cores if _torchrun_mpmd_selected() else None
    env = os.environ.copy()
    # Give every stage invocation its own compiler scratch dir. The vendor
    # ModelBuilder.trace() starts by rmtree-ing its workdir, and the default
    # layout keys scratch by COMPONENT name only ("/tmp/nxd_model/transformer"),
    # so two concurrent difflet compiles of different models would delete each
    # other's in-flight scratch mid-compile (observed: a HunyuanVideo compile
    # killed a running Wan compile with an NCC internal error). Compiled
    # artifacts and the neuronx-cc result cache are unaffected — this dir is
    # pure scratch. Honor an explicit caller override.
    if "BASE_COMPILE_WORK_DIR" not in os.environ:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", orchestrator.split("/")[-1]).lower()
        env["BASE_COMPILE_WORK_DIR"] = f"/tmp/nxd_model/{slug}-{stage}-{uuid.uuid4().hex[:8]}/"
    if nproc is not None:
        _prepare_mpmd_env(env, num_cores, strict_environment=strict_environment)
    elif strict_environment:
        visible_core_ids = select_neuron_core_ids(required_num_cores=num_cores)
        env["NEURON_RT_VISIBLE_CORES"] = ",".join(str(core_id) for core_id in visible_core_ids)
        if not _is_whole_device(num_cores):
            env["NEURON_RT_NUM_CORES"] = str(num_cores)
        elif "NEURON_RT_NUM_CORES" in env:
            # the inherited pin would be rejected even though it names the
            # whole device; the default allocation is what it describes anyway
            env.pop("NEURON_RT_NUM_CORES", None)
        if virtual_core_size is None:
            env.pop("NEURON_RT_VIRTUAL_CORE_SIZE", None)
        else:
            env["NEURON_RT_VIRTUAL_CORE_SIZE"] = str(virtual_core_size)
        env.pop("NEURON_LOGICAL_NC_CONFIG", None)
        env.update({"WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"})
    else:
        if not _is_whole_device(num_cores):
            env.setdefault("NEURON_RT_NUM_CORES", str(num_cores))
        if virtual_core_size is not None:
            env.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", str(virtual_core_size))

    cmd = build_stage_command(orchestrator, stage, cli_args, nproc=nproc)
    try:
        if nproc is None:
            subprocess.run(cmd, env=env, check=True)
        else:
            _run_torchrun(cmd, env)
    except BaseException:
        # Keep the scratch dir for post-mortem on failure.
        raise
    else:
        scratch = env.get("BASE_COMPILE_WORK_DIR")
        if scratch and scratch != os.environ.get("BASE_COMPILE_WORK_DIR"):
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)
