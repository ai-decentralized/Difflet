from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "difflet" / "offline" / "cache_profile"
ALLOWED_SCRIPT_IMPORTS = {
    "scripts.flux_cache_execution_policy",
    "scripts.flux_cache_natural_range_gate",
    "scripts.flux_cache_protocol",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_offline_profile_package_only_imports_the_shared_runtime_profile_contract():
    imports = set().union(*(_imports(path) for path in PACKAGE.glob("*.py")))

    assert {
        name
        for name in imports
        if name.startswith(("difflet.models", "difflet.pipeline", "difflet.serving"))
    } == {"difflet.pipeline.cache", "difflet.pipeline.cache.profile"}


def test_offline_profile_package_has_an_explicit_stable_script_boundary():
    imports = set().union(*(_imports(path) for path in PACKAGE.glob("*.py")))
    script_imports = {name for name in imports if name.startswith("scripts.")}

    assert script_imports == ALLOWED_SCRIPT_IMPORTS


def test_user_facing_commands_are_thin_compatibility_wrappers():
    wrappers = {
        ROOT / "scripts" / "build_flux_cache_profile.py": (
            "difflet.offline.cache_profile.builder"
        ),
        ROOT / "scripts" / "derive_flux_cache_schedule.py": (
            "difflet.offline.cache_profile.derivation"
        ),
    }

    for path, expected_import in wrappers.items():
        assert _imports(path) == {expected_import}
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 10
