"""Import hygiene for the GPU flow.

No vendor SDK or sibling-backend import is allowed under ``difflet/backends/cuda/``,
``difflet/backends/gpu/``, the shared ``difflet/backends/_torch_megatron/``, the GPU
platforms, or the backend-neutral model files the GPU flow runs. A convention would
drift; a scan does not.

The last test checks the CUDA ops modules provide every name ``difflet.ops`` exports.
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

# Directories are scanned recursively; a file entry is scanned on its own (the GPU
# platforms, and the backend-neutral model files the GPU flow runs).
SCANNED = [
    Path("difflet/backends/cuda"),
    Path("difflet/backends/gpu"),
    Path("difflet/backends/_torch_megatron"),
    Path("difflet/platforms/base.py"),
    Path("difflet/platforms/gpu.py"),
    Path("difflet/platforms/cuda.py"),
    Path("difflet/platforms/rocm.py"),
    Path("difflet/models/spec.py"),
    Path("difflet/models/flux/modeling_flux.py"),
    Path("difflet/models/flux/dit_config.py"),
    Path("difflet/models/flux/contract.py"),
    Path("difflet/models/flux/spec.py"),
    Path("difflet/models/qwen_image/spec.py"),
    Path("difflet/models/wan/dit_config.py"),
    Path("difflet/models/wan/spec.py"),
    Path("difflet/models/hunyuan_video/modeling_hunyuan_video.py"),
    Path("difflet/models/hunyuan_video/spec.py"),
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


@pytest.mark.parametrize("module", OPS_MODULES)
def test_cuda_ops_impl_provides_the_ops_contract(module):
    from difflet.ops import _EXPORTS

    mod = importlib.import_module(f"difflet.backends.cuda.ops_impl.{module}")
    expected = {name for name, (m, _) in _EXPORTS.items() if m == module}
    assert expected and expected <= set(mod.__all__), module
    for name in expected:
        assert hasattr(mod, name), name
