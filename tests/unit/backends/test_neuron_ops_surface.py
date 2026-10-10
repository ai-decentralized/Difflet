"""The neuron backend must export every name the trainium backend exports.

Text-based like ``test_tpu_collectives.py``: ``__all__`` is read from the source
of each ``ops_impl`` module, so the check needs no backend import. ``PENDING``
lists the names whose neuron module has not landed yet; it shrinks as they land
(C4 linear, C6 mx) and ends as ``set()``.
"""

import re
from pathlib import Path

import pytest

import difflet
import difflet.ops as ops
from difflet.backends import registry

BACKENDS = Path(difflet.__file__).resolve().parent / "backends"

PENDING = {
    "ColumnParallelLinear",
    "ParallelEmbedding",
    "RowParallelLinear",
    "dequantize_mx",
    "linear_mx",
    "matmul_mx",
    "quantize_mx",
}


def exports(backend):
    out = set()
    for path in (BACKENDS / backend / "ops_impl").glob("*.py"):
        text = path.read_text()
        first = re.search(r"__all__\s*=\s*\[(.*?)\]", text, re.S)
        if first:
            out |= set(re.findall(r'"([^"]+)"', first.group(1)))
        for extra in re.finditer(r"__all__\s*\+=\s*\[(.*?)\]", text, re.S):
            out |= set(re.findall(r'"([^"]+)"', extra.group(1)))
    return out


def test_neuron_covers_the_trainium_ops_surface():
    missing = exports("trainium") - exports("neuron")
    assert missing == PENDING, (
        f"missing beyond PENDING: {sorted(missing - PENDING)}; "
        f"landed, drop from PENDING: {sorted(PENDING - missing)}"
    )


@pytest.mark.parametrize("name", sorted(set(ops.__all__) - PENDING))
def test_every_landed_ops_name_resolves_for_neuron(monkeypatch, name):
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    registry._get_backend_by_name.cache_clear()
    try:
        assert getattr(ops, name) is not None
    finally:
        registry._get_backend_by_name.cache_clear()
