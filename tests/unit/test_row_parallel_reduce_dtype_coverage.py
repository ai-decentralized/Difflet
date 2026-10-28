"""Every RowParallelLinear in difflet states its all-reduce dtype.

NxD's RowParallelLinear defaults to reduce_dtype=torch.float32, so a call site
that omits the argument silently all-reduces in fp32. MiniMax-H3 keeps the
default on purpose; backend ops_impl passthroughs are not model call sites.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_ALLOWED_PREFIXES = ("difflet/backends/trainium/minimax_h3/",)

# Sites still on the NxD fp32 default, by file (MiniMax-H3 aside): none.
EXPECTED_DEFAULT_SITES: dict[str, int] = {}


def _default_sites() -> Counter:
    sites: Counter = Counter()
    for path in sorted((ROOT / "difflet").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if "/ops_impl/" in rel or rel.startswith(_ALLOWED_PREFIXES):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "RowParallelLinear":
                continue
            keywords = {kw.arg for kw in node.keywords}
            if "reduce_dtype" not in keywords and None not in keywords:
                sites[rel] += 1
    return sites


def test_row_parallel_sites_state_reduce_dtype():
    assert dict(_default_sites()) == EXPECTED_DEFAULT_SITES
