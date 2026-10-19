"""``Fp32GELU`` (CPU only): the neuron backend's feed-forward activation in fp32, a module of its
own in place of diffusers' ``GELU`` (``difflet/backends/neuron/core/activations.py``).

LTX-2's FFNs used to get it by swapping the class of diffusers' ``GELU`` at runtime
(``act.__class__ = ...``). These pin that the module that replaced the swap computes the same
bits, keeps the ``ff.net.0.proj.*`` keys and the very same ``proj``, and behaves as an ordinary
module (deepcopy, pickle, no longer an ``isinstance`` of ``GELU``).
"""

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.models.activations import GELU  # noqa: E402
from diffusers.models.attention import FeedForward  # noqa: E402

from difflet.backends.neuron.core.activations import Fp32GELU, use_fp32_gelu  # noqa: E402


def _feed_forward(seed: int = 0) -> FeedForward:
    torch.manual_seed(seed)
    return FeedForward(16, activation_fn="gelu-approximate").to(torch.bfloat16)


class _SwappedGELU(GELU):
    """The runtime class swap this module replaced, reproduced for the comparison."""

    def gelu(self, gate):
        return torch.nn.functional.gelu(gate.float(), approximate=self.approximate).to(gate.dtype)


def test_same_bits_as_the_class_swap_and_fp32_inside():
    reference, replaced = _feed_forward(), _feed_forward()
    reference.net[0].__class__ = _SwappedGELU
    assert use_fp32_gelu(replaced)
    x = torch.randn(3, 5, 16, generator=torch.Generator().manual_seed(1)).mul(3).to(torch.bfloat16)
    with torch.no_grad():
        torch.testing.assert_close(replaced(x), reference(x), atol=0, rtol=0)
        gate = replaced.net[0].proj(x)
        expected = torch.nn.functional.gelu(gate.float(), approximate="tanh").to(torch.bfloat16)
        torch.testing.assert_close(replaced.net[0](x), expected, atol=0, rtol=0)
        # (The CPU's bf16 GELU already computes in fp32, so only the device tells the two
        # apart; the bits above are what the device path must reproduce.)


def test_keeps_the_projection_and_the_state_dict_keys():
    ff = _feed_forward()
    proj, keys = ff.net[0].proj, list(ff.state_dict())
    assert use_fp32_gelu(ff)
    activation = ff.net[0]
    assert type(activation) is Fp32GELU and not isinstance(activation, GELU)
    assert activation.proj is proj and activation.approximate == "tanh"
    assert list(ff.state_dict()) == keys and "net.0.proj.weight" in keys
    assert not use_fp32_gelu(ff)  # already replaced: unchanged
    exact = FeedForward(16, activation_fn="geglu")
    assert not use_fp32_gelu(exact) and type(exact.net[0]).__name__ == "GEGLU"


def test_copies_and_pickles_as_a_plain_module():
    ff = _feed_forward()
    use_fp32_gelu(ff)
    for clone in (copy.deepcopy(ff), pickle.loads(pickle.dumps(ff))):
        assert type(clone.net[0]) is Fp32GELU
        x = torch.ones(1, 2, 16, dtype=torch.bfloat16)
        with torch.no_grad():
            torch.testing.assert_close(clone(x), ff(x), atol=0, rtol=0)
