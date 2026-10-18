"""The LTX-2 modeling module is backend-neutral: it imports no backend and no NxD, and the
Trainium transformer module re-exports its classes unchanged."""

import ast
from pathlib import Path

_MODELING = Path(__file__).resolve().parents[4] / "difflet/models/ltx_2/modeling_ltx_2.py"
_FORBIDDEN = ("difflet.backends", "neuronx_distributed", "torch_neuronx", "torch_xla")


def _imported_modules(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_modeling_ltx_2_imports_no_backend():
    imported = _imported_modules(_MODELING)
    assert "difflet.ops" in imported
    offenders = [name for name in imported if name.startswith(_FORBIDDEN)]
    assert offenders == []


def test_trainium_transformer_reexports_the_modeling_classes():
    trainium = _MODELING.parents[2] / "backends/trainium/ltx_2/transformer.py"
    tree = ast.parse(trainium.read_text())
    reexported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "difflet.models.ltx_2.modeling_ltx_2"
        for alias in node.names
    }
    defined = {
        node.name
        for node in ast.walk(ast.parse(_MODELING.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.col_offset == 0
    }
    assert defined <= reexported
    # Nothing the modeling module defines is defined again on the Trainium side.
    trainium_defined = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert not (trainium_defined & defined)
