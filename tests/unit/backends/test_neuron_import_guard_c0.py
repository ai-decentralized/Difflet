"""C0 import guard: the neuron backend's import surface never reaches torch_xla or NxD.

Each module is imported in a fresh interpreter with ``DIFFLET_BACKEND=neuron``, so
the assertion is about what *that module* pulls in, not about whatever this pytest
process imported earlier.

The child blocks the XLA / NxD stack in two layers; both are needed on Python 3.14:

* ``sys.modules[name] = None`` for the top-level names. ``import torch_xla`` then
  raises ``ModuleNotFoundError``, while ``importlib.util.find_spec("torch_xla")``
  returns ``None``, which is what the import-time availability probes in diffusers
  (``utils/import_utils.py:193``) and transformers (``utils/import_utils.py:368``)
  expect from an absent package. A finder that raised there would crash them.
* a ``find_spec`` meta-path finder that refuses every other ``torch_xla*`` /
  ``neuronx_distributed*`` name and records the attempt (a ``find_module`` finder
  is ignored on Python 3.14).

Both layers also make the test meaningful in a venv where those packages ARE
installed (the Trainium venv): an import attempt fails instead of succeeding.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_GUARD_PROGRAM = r'''
import importlib
import importlib.abc
import sys

PREFIXES = ("torch_xla", "neuronx_distributed")
SENTINELS = ("torch_xla", "neuronx_distributed", "neuronx_distributed_inference")


def is_blocked(name):
    return name.partition(".")[0].startswith(PREFIXES)


class BlockXlaAndNxd(importlib.abc.MetaPathFinder):
    refused = []

    def find_spec(self, fullname, path=None, target=None):
        if is_blocked(fullname):
            self.refused.append(fullname)
            raise ModuleNotFoundError(
                f"{fullname} is blocked by the neuron import guard", name=fullname
            )
        return None


sys.meta_path.insert(0, BlockXlaAndNxd())
for name in SENTINELS:
    sys.modules[name] = None

importlib.import_module(sys.argv[1])

loaded = sorted(n for n, m in sys.modules.items() if is_blocked(n) and m is not None)
assert not loaded, f"blocked modules were loaded: {loaded}"
assert not BlockXlaAndNxd.refused, f"blocked imports were attempted: {BlockXlaAndNxd.refused}"
neuronx = sys.modules.get("torch_neuronx")
if neuronx is not None:
    assert not neuronx.is_neuron_runtime_initialized(), "import initialized the Neuron runtime"
print("IMPORT-GUARD-OK", sys.argv[1])
'''

_NEEDS_C4 = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="needs neuron linear (C4): difflet.ops.ColumnParallelLinear resolves to "
    "difflet.backends.neuron.ops_impl.linear",
)

GUARDED_MODULES = [
    "difflet.ops",
    "difflet.pipeline",
    "difflet.models.wan.vae",
    pytest.param("difflet.models.wan.umt5.modeling_umt5", marks=_NEEDS_C4),
    pytest.param("difflet.models.wan.modeling_wan", marks=_NEEDS_C4),
]


def _run_guard(module: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["DIFFLET_BACKEND"] = "neuron"
    env["DIFFLET_DISABLE_PREWARM"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p
    )
    return subprocess.run(
        [sys.executable, "-c", _GUARD_PROGRAM, module],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=REPO_ROOT,
    )


def _last_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


@pytest.mark.parametrize("module", GUARDED_MODULES)
def test_module_imports_under_neuron_without_xla_or_nxd(module):
    result = _run_guard(module)
    assert result.returncode == 0, (
        f"importing {module} with DIFFLET_BACKEND=neuron failed: {_last_line(result.stderr)}\n"
        f"{result.stderr[-4000:]}"
    )
    assert f"IMPORT-GUARD-OK {module}" in result.stdout


@pytest.mark.parametrize(
    "module, message",
    [
        ("torch_xla", "import of torch_xla halted; None in sys.modules"),
        (
            "neuronx_distributed.parallel_layers",
            "No module named 'neuronx_distributed.parallel_layers'",
        ),
        ("neuronx_distributed_training", "blocked by the neuron import guard"),
    ],
)
def test_guard_refuses_xla_and_nxd(module, message):
    # The guard itself must bite, whether or not the package is installed here.
    result = _run_guard(module)
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert message in result.stderr
