"""Unit tests for the neuron backend's MX ops (always unsupported); no Neuron hardware needed."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from difflet.backends.neuron.ops_impl import mx as neuron_mx  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MX_NAMES = ("dequantize_mx", "linear_mx", "matmul_mx", "quantize_mx")
MESSAGE = (
    "MX (microscaling) quantization is not supported on the neuron backend (TorchNeuron); "
    "run without MX quantization or use DIFFLET_BACKEND=trainium"
)


def _expected(name):
    return f"{name}: {MESSAGE}"


@pytest.fixture
def neuron_selected(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")


def _wrapper_call(name):
    """Arguments shaped like a real call through the ``difflet.ops.mx`` wrappers."""
    x = torch.zeros(8, 4, dtype=torch.bfloat16)
    data = torch.zeros(8, 1, dtype=torch.uint32)
    scale = torch.zeros(1, 1, dtype=torch.uint8)
    act = torch.zeros(128, 512, dtype=torch.bfloat16)
    weight = torch.zeros(512, 512, dtype=torch.bfloat16)
    return {
        "quantize_mx": ((x,), {"group_size": 32}),
        "dequantize_mx": ((data, scale), {"output_dtype": torch.float32}),
        "matmul_mx": ((data, scale, data, scale), {"accumulate": True}),
        "linear_mx": ((act, weight), {"out_dtype": torch.bfloat16}),
    }[name]


def test_exports_exactly_the_four_mx_functions():
    assert neuron_mx.__all__ == ["dequantize_mx", "linear_mx", "matmul_mx", "quantize_mx"]


def test_message_names_the_backend_and_the_ways_out():
    assert neuron_mx._UNSUPPORTED == MESSAGE


@pytest.mark.parametrize("name", MX_NAMES)
def test_backend_function_raises_for_any_arguments(name):
    fn = getattr(neuron_mx, name)
    for args, kwargs in [((), {}), ((torch.zeros(8, 4),), {"dtype": "float8_e4m3fn_x4"})]:
        with pytest.raises(NotImplementedError) as info:
            fn(*args, **kwargs)
        assert str(info.value) == _expected(name)


@pytest.mark.parametrize("name", MX_NAMES)
def test_difflet_ops_resolves_to_the_neuron_function(name, neuron_selected):
    import difflet.ops

    fn = getattr(difflet.ops, name)
    assert fn is getattr(neuron_mx, name)
    with pytest.raises(NotImplementedError) as info:
        fn()
    assert str(info.value) == _expected(name)


@pytest.mark.parametrize("name", MX_NAMES)
def test_difflet_ops_mx_wrapper_raises(name, neuron_selected):
    from difflet.ops import mx as ops_mx

    args, kwargs = _wrapper_call(name)
    with pytest.raises(NotImplementedError) as info:
        getattr(ops_mx, name)(*args, **kwargs)
    assert str(info.value) == _expected(name)


def test_covers_the_trainium_mx_surface():
    # Text-based: trainium's mx module cannot import in this venv (nkilib.core.utils.tensor_view).
    text = (REPO / "difflet/backends/trainium/ops_impl/mx.py").read_text()
    block = re.search(r"__all__\s*=\s*\[(.*?)\]", text, re.S).group(1)
    assert set(re.findall(r'"([^"]+)"', block)) == set(neuron_mx.__all__)


_FRESH_PROCESS_PROBE = textwrap.dedent(
    """
    import sys

    import difflet.ops as ops
    from difflet.ops import mx as ops_mx

    for name in ("dequantize_mx", "linear_mx", "matmul_mx", "quantize_mx"):
        module = getattr(ops, name).__module__
        assert module == "difflet.backends.neuron.ops_impl.mx", (name, module)
    try:
        ops_mx.quantize_mx(None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("difflet.ops.mx.quantize_mx did not raise on neuron")
    # `import torch` autoloads torch_neuronx (and nki, nkilib) on this host, so only
    # the trainium MX stack and the XLA/NxD packages are forbidden here.
    forbidden = ("difflet.backends.trainium", "torch_xla", "neuronx_distributed")
    loaded = sorted(m for m in sys.modules if m.startswith(forbidden))
    assert not loaded, loaded
    """
)


def test_resolving_mx_never_loads_the_trainium_or_xla_stack():
    env = {**os.environ, "DIFFLET_BACKEND": "neuron", "PYTHONPATH": str(REPO)}
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_PROBE],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
