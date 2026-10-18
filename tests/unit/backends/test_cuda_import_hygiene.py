"""Import hygiene for the CUDA backend (CUDA plan Phase 1, task 1.4).

The plan's Global Constraints forbid any vendor SDK or sibling-backend import
under ``difflet/backends/cuda/`` (and, from Phase 2, the shared
``difflet/backends/_torch_megatron/``). A convention would drift; a scan does not.

The second test lists every ``difflet.ops`` module the CUDA backend must provide.
They are ``xfail(strict=True)`` until Phase 2 lands each one, so landing a module
flips its test to an unexpected pass and the executor must remove the marker —
progress that cannot be forgotten.
"""

import ast
import importlib
from pathlib import Path

import pytest

FORBIDDEN_PREFIXES = (
    "neuronx_distributed",
    "torch_neuronx",
    "torch_xla",
    "nkilib",
    "difflet.backends.trainium",
    "difflet.backends.tpu",
)

# Directories are scanned recursively; a file entry is scanned on its own (the
# backend-neutral model files the CUDA backend runs, F5 Task 2).
SCANNED = [
    Path("difflet/backends/cuda"),
    Path("difflet/backends/_torch_megatron"),
    Path("difflet/models/flux/modeling_flux.py"),
    Path("difflet/models/flux/dit_config.py"),
    Path("difflet/models/flux/cuda_application.py"),
    Path("difflet/models/flux/contract.py"),
    Path("difflet/models/wan/dit_config.py"),
    Path("difflet/models/wan/cuda_application.py"),
]

OPS_MODULES = ("attention", "collectives", "embeddings", "linear", "mx", "norm", "platform")


def _is_forbidden(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in FORBIDDEN_PREFIXES)


def scan_forbidden_imports(root: Path) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name):
                        violations.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _is_forbidden(module):
                    violations.append((path, node.lineno, module))
                elif module == "difflet.backends":
                    for alias in node.names:
                        if _is_forbidden(f"{module}.{alias.name}"):
                            violations.append((path, node.lineno, f"{module}.{alias.name}"))
    return violations


def test_cuda_backend_has_no_forbidden_imports():
    repo = Path(__file__).resolve().parents[3]
    missing = [root for root in SCANNED if not (repo / root).exists()]
    assert missing == [], f"SCANNED entries that do not exist: {missing}"
    violations = [v for root in SCANNED for v in scan_forbidden_imports(repo / root)]
    assert violations == [], violations


def test_scan_detects_a_forbidden_import(tmp_path):
    (tmp_path / "bad.py").write_text("from difflet.backends.trainium.core import config\n")
    assert scan_forbidden_imports(tmp_path) == [
        (tmp_path / "bad.py", 1, "difflet.backends.trainium.core")
    ]


def test_scan_accepts_a_single_file(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import torch\nimport torch_xla.core.xla_model as xm\n")
    assert scan_forbidden_imports(bad) == [(bad, 2, "torch_xla.core.xla_model")]


# Modules Phase 2 has not landed yet: their import is expected to fail (strict), so
# landing one flips its test and the executor must remove it from this set.
_PENDING: set[str] = set()  # every module landed in PR 3 Tasks 2-4


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            m,
            marks=(
                pytest.mark.xfail(strict=True, reason="lands in Phase 2 (PR 3)")
                if m in _PENDING
                else ()
            ),
        )
        for m in OPS_MODULES
    ],
)
def test_cuda_ops_impl_module_imports(module):
    importlib.import_module(f"difflet.backends.cuda.ops_impl.{module}")
