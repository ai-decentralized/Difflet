from __future__ import annotations

import os
import subprocess
import sys

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
    subprocess.run(cmd, env=env, check=True)
