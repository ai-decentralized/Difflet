from __future__ import annotations

import os
import subprocess
import sys


def run_stage(
    orchestrator: str,
    stage: str,
    num_cores: int,
    virtual_core_size: int | None,
    cli_args: list[str],
) -> None:
    """Spawn a stage subprocess with the correct Neuron env vars.

    Uses setdefault so any value already set in the user's shell wins.
    """
    env = os.environ.copy()
    env.setdefault("NEURON_RT_NUM_CORES", str(num_cores))
    if virtual_core_size is not None:
        env.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", str(virtual_core_size))

    cmd = [
        sys.executable, "-m", "difflet.cli.stage",
        "--orchestrator", orchestrator,
        "--stage", stage,
        *cli_args,
    ]
    subprocess.run(cmd, env=env, check=True)
