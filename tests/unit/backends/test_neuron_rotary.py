"""The neuron backend's ``apply_rotary_emb`` (a pair swap by matmul instead of stride-2 slices)
is bit-identical to the CPU backend's interleaved-pair op, the reference every Wan test uses."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from difflet.backends.cpu.ops_impl.embeddings import apply_rotary_emb as cpu_rope  # noqa: E402
from difflet.backends.neuron.ops_impl.embeddings import apply_rotary_emb as neuron_rope  # noqa: E402


def _freqs(seq: int, dim: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Wan-style tables: ``[S, D]`` fp32 with each pair's angle on both of its positions."""
    theta = torch.rand(seq, dim // 2, generator=gen) * 6.283
    cos = theta.cos().repeat_interleave(2, dim=-1)
    sin = theta.sin().repeat_interleave(2, dim=-1)
    return cos, sin


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 4, 7, 16), (2, 3, 5, 128)])
def test_neuron_rope_is_bit_identical_to_the_cpu_op(dtype, shape):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(*shape, generator=gen).to(dtype)
    cos, sin = _freqs(shape[-2], shape[-1], gen)
    ours, ref = neuron_rope(x, cos, sin), cpu_rope(x, cos, sin)
    assert ours.dtype == ref.dtype == dtype
    assert torch.equal(ours, ref)


def test_neuron_rope_rotates_each_pair():
    """A hand-checked pair: (x0, x1) -> (x0 c - x1 s, x0 s + x1 c)."""
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos = torch.tensor([[0.0, 0.0, 1.0, 1.0]])  # pair 0 at 90 degrees, pair 1 untouched
    sin = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    out = neuron_rope(x, cos, sin)
    assert torch.equal(out, torch.tensor([[[[-2.0, 1.0, 3.0, 4.0]]]]))
