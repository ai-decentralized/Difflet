"""Keep a CPU reference process away from the Neuron device. Import FIRST, before torch.

Why: in the NxD venv, diffusers / accelerate / transformers import ``torch_xla`` when it is
installed, which initializes ``torch_neuronx`` (PJRT) and makes the pipelines call
``xm.mark_step()`` every step; in the native venv, ``import torch`` autoloads
``torch_neuronx`` through the ``torch.backends`` entry point. Neither may happen here.

* ``USE_TORCH_XLA=0`` (accelerate, transformers) and ``TORCH_DEVICE_BACKEND_AUTOLOAD=0``
  (torch's backend autoload) are set before torch is imported;
* ``install()`` sets diffusers' ``_torch_xla_available`` to False (its pipelines then skip
  ``import torch_xla`` and ``xm.mark_step``) and then puts a finder first on ``sys.meta_path``
  that refuses every import of the Neuron / XLA packages;
* ``report()`` lists any such module that got loaded anyway and any open fd whose target
  mentions ``neuron`` (``/dev/neuron*``); the scripts fail if either is non-empty.
"""

from __future__ import annotations

import os
import sys

os.environ["USE_TORCH_XLA"] = "0"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

BLOCKED = (
    "torch_xla",
    "torch_neuronx",
    "libneuronxla",
    "neuronx_distributed",
    "neuronx_distributed_inference",
    "torchtitan_neuron",
)


class _NeuronImportBlocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name}: blocked in a CPU-only reference process (cpu_only.py)")
        return None


def install() -> None:
    import diffusers.utils.import_utils as iu

    iu._torch_xla_available = False
    if not any(isinstance(f, _NeuronImportBlocker) for f in sys.meta_path):
        sys.meta_path.insert(0, _NeuronImportBlocker())


def report() -> dict:
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    fds = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            tgt = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if "neuron" in tgt:
            fds.append(tgt)
    import diffusers.utils.import_utils as iu

    return {
        "blocked_modules_loaded": loaded,
        "neuron_fds_open": fds,
        "diffusers_torch_xla_available": bool(iu._torch_xla_available),
        "env": {k: os.environ.get(k) for k in ("USE_TORCH_XLA", "TORCH_DEVICE_BACKEND_AUTOLOAD")},
        "ok": not loaded and not fds and not iu._torch_xla_available,
    }
