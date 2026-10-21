from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import uuid
from collections.abc import Mapping

from difflet.cli.parent_death import die_with_parent
from difflet.common.neuron_cores import (
    resolve_available_neuron_core_ids,
    select_neuron_core_ids,
)


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
    """
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
    if strict_environment:
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

    cmd = [
        sys.executable,
        "-m",
        "difflet.cli.stage",
        "--orchestrator",
        orchestrator,
        "--stage",
        stage,
        *cli_args,
    ]
    try:
        subprocess.run(cmd, env=env, check=True)
    except BaseException:
        # Keep the scratch dir for post-mortem on failure.
        raise
    else:
        scratch = env.get("BASE_COMPILE_WORK_DIR")
        if scratch and scratch != os.environ.get("BASE_COMPILE_WORK_DIR"):
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)


def run_torchrun(world_size: int, argv: list[str], env: Mapping[str, str] | None = None) -> int:
    """Launch ``world_size`` local ranks of ``python <argv>`` under ``torchrun``; return its code.

    The CUDA backend's multi-process launch: ``torch.distributed.run --standalone``
    gives every rank the env rendezvous (``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK``,
    ``MASTER_ADDR/PORT``) that ``CudaBackend.prepare_runtime`` joins. ``argv`` is the module
    invocation, ``["-m", "difflet.cli.main", command, *args]``: torchrun's own ``-m`` makes
    each rank run it as ``python -m``. The return code is the caller's to judge.

    On Linux torchrun gets SIGTERM when this process dies (``timeout``, a SIGTERM, the DP
    router's own death signal to a cfg worker): its elastic agent then terminates the
    ranks, which would otherwise be orphaned holding their GPUs. SIGTERM, not SIGKILL —
    a SIGKILLed agent cannot pass the signal on. An unset ``OMP_NUM_THREADS`` becomes the
    rank's share of the host cores: torchrun would pin each rank to one thread, and the
    Wan loop keeps the scheduler and the fp32 latents on the host.
    """
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(world_size),
        *argv,
    ]
    child_env = dict(os.environ) if env is None else dict(env)
    if "OMP_NUM_THREADS" not in child_env:
        child_env["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 1) // world_size))
    preexec = die_with_parent(signal.SIGTERM) if sys.platform.startswith("linux") else None
    return subprocess.run(cmd, env=child_env, check=False, preexec_fn=preexec).returncode
