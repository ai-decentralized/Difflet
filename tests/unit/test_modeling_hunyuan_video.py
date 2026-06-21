"""Unit tests for difflet.models.hunyuan_video.modeling_hunyuan_video.

Forward / parity coverage lives in
``tests/unit/test_hunyuan_video_attention.py`` and the registry checks in
``tests/unit/test_hunyuan_video_registration.py``; this file is the
per-model AST import guard, mirroring ``tests/unit/test_modeling_wan.py``.
The repo-wide guard at ``scripts/test_imports.sh`` already rejects
``difflet.core`` and the four removed ``difflet.utils`` paths; the per-file
AST walk below keeps the constraint visible inside the modeling code's
own neighborhood.
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_modeling_hunyuan_video_imports_only_from_allowed_modules():
    src = Path("/home/ubuntu/difflet/difflet/models/hunyuan_video/modeling_hunyuan_video.py").read_text()
    tree = ast.parse(src)
    forbidden_roots = {"neuronx_distributed", "nkilib", "torch_neuronx"}
    forbidden_prefixes = ("difflet.core",)
    offending: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in forbidden_roots or alias.name.startswith(forbidden_prefixes):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            if root in forbidden_roots or mod.startswith(forbidden_prefixes):
                offending.append(mod)

    assert not offending, f"forbidden imports in modeling_hunyuan_video.py: {offending}"
