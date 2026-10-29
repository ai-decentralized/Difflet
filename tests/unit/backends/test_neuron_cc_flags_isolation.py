"""A Wan application built for the neuron device appends the ccop overlap flag to this process's
NEURON_CC_FLAGS (difflet/backends/neuron/wan/compile_options.py), as it must in a Wan process.
In the shared pytest process the unit conftest restores the variable after every test, so no later
test's device compile (another model's included) gets the flag. The two tests run in file order.
"""

import os

import pytest

torch = pytest.importorskip("torch")

_AMBIENT = os.environ.get("NEURON_CC_FLAGS")
CCOP = "--tensorizer-options=--enable-ccop-compute-overlap"
_HELD: list = []


def test_a_wan_application_on_the_device_sets_neuron_cc_flags(monkeypatch, tmp_path):
    from difflet.backends.neuron.wan.transformer import NeuronWanTransformerApplication
    from tests.unit.backends._neuron_wan_toy import tiny_wan_config

    monkeypatch.delenv("DIFFLET_NEURON_WAN_CCOP_OVERLAP", raising=False)
    # Held to the end of the test: the flag lives as long as a Wan application does; the
    # conftest drops the hold before restoring the variable.
    app = NeuronWanTransformerApplication(model_path=tmp_path, config=tiny_wan_config(),
                                          dtype=torch.bfloat16, device="neuron")
    assert CCOP in os.environ["NEURON_CC_FLAGS"].split()
    _HELD.append(app)  # still alive in the next test: its collection must not touch the env


def test_the_next_test_starts_with_the_ambient_value(monkeypatch):
    import gc

    assert os.environ.get("NEURON_CC_FLAGS") == _AMBIENT
    # The previous test's application, collected now, leaves this test's value alone.
    monkeypatch.setenv("NEURON_CC_FLAGS", f"{CCOP} -O1")
    _HELD.clear()
    gc.collect()
    assert os.environ["NEURON_CC_FLAGS"] == f"{CCOP} -O1"
