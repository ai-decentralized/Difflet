import ast
from pathlib import Path


def test_mx_public_surface_imports_only_dispatch_and_torch():
    tree = ast.parse(Path("difflet/ops/mx.py").read_text(), filename="difflet/ops/mx.py")
    allowed = {"__future__", "torch", "difflet.ops._dispatch"}
    seen = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            seen.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            seen.add(node.module or "")

    assert seen <= allowed
