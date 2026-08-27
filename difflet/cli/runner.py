from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid

from difflet.common.neuron_cores import select_neuron_core_ids


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
        env["NEURON_RT_NUM_CORES"] = str(num_cores)
        if virtual_core_size is None:
            env.pop("NEURON_RT_VIRTUAL_CORE_SIZE", None)
        else:
            env["NEURON_RT_VIRTUAL_CORE_SIZE"] = str(virtual_core_size)
        env.pop("NEURON_LOGICAL_NC_CONFIG", None)
        env.update({"WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"})
    else:
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
