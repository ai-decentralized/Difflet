from __future__ import annotations

import ast
import re
from pathlib import Path

import difflet.pipeline.cache as cache_api

_FORBIDDEN_IMPORT_PREFIXES = (
    "difflet.backends",
    "difflet.cli",
    "difflet.common",
    "difflet.models",
    "difflet.serving",
    "diffusers",
    "neuronx_distributed",
    "torch_neuronx",
)


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


def test_cache_core_has_no_host_or_hardware_imports():
    repository_root = Path(__file__).resolve().parents[4]
    cache_package = repository_root / "difflet" / "pipeline" / "cache"

    violations = []
    for path in sorted(cache_package.glob("*.py")):
        for module in _imported_modules(path):
            if module.startswith(_FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{path.name}: {module}")

    assert violations == []


def test_removed_controller_classes_are_not_public_api():
    assert not hasattr(cache_api, "CacheRuntimeController")
    assert not hasattr(cache_api, "CachePlanController")
    assert not hasattr(cache_api, "CachePlan")
    assert not hasattr(cache_api, "CacheMask")
    assert not hasattr(cache_api, "ResolvedCacheSession")
    assert cache_api.CacheSession.__name__ == "CacheSession"
    assert cache_api.TeaCacheControllerAdapter.__name__ == "TeaCacheControllerAdapter"


def test_cache_core_class_and_function_names_do_not_encode_schema_revisions():
    repository_root = Path(__file__).resolve().parents[4]
    cache_package = repository_root / "difflet" / "pipeline" / "cache"
    revision_suffix = re.compile(r"(?:_v|V)[12]$")
    violations = []
    for path in sorted(cache_package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if revision_suffix.search(node.name):
                    violations.append(f"{path.name}: {node.name}")

    assert violations == []
