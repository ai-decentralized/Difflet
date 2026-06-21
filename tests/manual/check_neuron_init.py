"""Manual smoke: verify Neuron runtime + torch_xla + Difflet imports on hardware.

Run on a Trainium machine:

    PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \\
    PYTHONPATH=/home/ubuntu/difflet \\
    torchrun --nproc_per_node=2 tests/manual/check_neuron_init.py

Does NOT trigger any model download or AOT compile. Only verifies:

  1. torch_neuronx + torch_xla initialize cleanly under torchrun
  2. Each rank gets its own xla_device
  3. A trivial tensor op materializes (cheap proxy that the Neuron runtime
     is actually wired through)
  4. Difflet top-level + Flux entry imports succeed in a multi-process context
"""

from __future__ import annotations

import os
import sys
import time


def _log(msg: str) -> None:
    rank = os.environ.get("RANK", "?")
    print(f"[rank {rank}] {msg}", flush=True)


def main() -> int:
    _log("starting...")
    t0 = time.monotonic()

    # 1. Neuron + XLA init
    import torch
    import torch_neuronx  # noqa: F401  (init side-effect)
    import torch_xla.core.xla_model as xm
    _log(f"torch={torch.__version__} torch_xla loaded ({time.monotonic() - t0:.1f}s)")

    device = xm.xla_device()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    _log(f"world={world} xla_device={device}")

    # 2. Difflet import smoke (uses lazy __getattr__ + factory string)
    from difflet import DiffletPipeline, DiffletParallelConfig, register_model  # noqa: F401
    from difflet.registry import registered_models, resolve_model
    _log(f"difflet top-level import OK; registered = {[m.name for m in registered_models()]}")

    # Resolve the Flux registry entry (does not call its factory).
    flux_entry = resolve_model("black-forest-labs/FLUX.1-dev")
    _log(f"flux entry default_parallel = {flux_entry.default_parallel}")

    # 3. Heavy import — the Flux Neuron application module. This pulls in
    # NXD parallel layers + nkilib + custom kernels. If anything is broken on
    # the hardware path this is where we'll see it.
    from difflet.models.flux.application import NeuronFluxApplication  # noqa: F401
    _log("Flux Neuron application module imported")

    # 4. Trivial XLA op — proves the runtime can lower & execute something.
    t = torch.tensor([float(rank)], device=device)
    out = (t * 2.0 + 1.0)
    xm.mark_step()
    value = out.cpu().item()
    _log(f"xla op result: rank*2+1 = {value} (expected {rank * 2 + 1})")
    assert value == rank * 2 + 1, "XLA op produced wrong value!"

    _log(f"OK ({time.monotonic() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
