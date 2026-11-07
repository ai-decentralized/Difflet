"""The neuron backend's rotary ops (a pair swap by matmul instead of stride-2 slices) are
bit-identical to the laws they replace: ``apply_rotary_emb`` to the CPU backend's interleaved-pair
op, the reference every Wan test uses, and ``apply_rotary_emb_fp32`` to diffusers'
``apply_rotary_emb(use_real=True, use_real_unbind_dim=-1)``, the law of FLUX's call sites."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from diffusers.models.embeddings import apply_rotary_emb as diffusers_rope  # noqa: E402

from difflet.backends._torch_megatron.embeddings import (  # noqa: E402
    apply_rotary_emb_fp32 as reference_rope_fp32,
)
from difflet.backends.cpu.ops_impl.embeddings import apply_rotary_emb as cpu_rope  # noqa: E402
from difflet.backends.neuron.ops_impl.embeddings import (  # noqa: E402
    apply_rotary_emb as neuron_rope,
    apply_rotary_emb_fp32 as neuron_rope_fp32,
)


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


@pytest.mark.parametrize("op", [neuron_rope_fp32, reference_rope_fp32], ids=["neuron", "reference"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sequence_dim", [1, 2])
@pytest.mark.parametrize("d", [16, 128])
def test_fp32_rope_is_bit_identical_to_diffusers(op, dtype, table_dtype, sequence_dim, d):
    """Every input the call sites feed: bf16 or fp32 activations and tables, the ``[B, H, S, D]``
    and ``[B, S, H, D]`` layouts, and random tables, so no pair's two positions share a value
    (the law must not lean on the tables FLUX happens to build)."""
    gen = torch.Generator().manual_seed(0)
    shape = (2, 3, 7, d) if sequence_dim == 2 else (2, 7, 3, d)
    x = torch.randn(*shape, generator=gen).to(dtype)
    cos = torch.randn(7, d, generator=gen).to(table_dtype)
    sin = torch.randn(7, d, generator=gen).to(table_dtype)
    expect = diffusers_rope(x, (cos, sin), sequence_dim=sequence_dim)
    # The broadcast diffusers applies for each sequence_dim; the op takes the tables broadcast.
    c, s = (t[None, None] if sequence_dim == 2 else t[None, :, None] for t in (cos, sin))
    out = op(x, c, s)
    assert out.dtype == dtype and torch.equal(out, expect)


def test_every_other_backend_exports_the_reference_fp32_rope():
    """Only the neuron backend owns the op: the others trace diffusers' own lines, so the
    Trainium AoT FLUX graph is the one it compiled before the op existed."""
    for backend in ("cpu", "cuda", "tpu", "trainium"):
        module = importlib.import_module(f"difflet.backends.{backend}.ops_impl.embeddings")
        assert module.apply_rotary_emb_fp32 is reference_rope_fp32, backend
        assert "apply_rotary_emb_fp32" in module.__all__, backend


def test_flux_rotary_reaches_the_neuron_op_on_the_neuron_backend():
    """No CPU number tells the two ops apart, so the binding is checked by identity: FLUX's call
    sites (``difflet.layers.embeddings.apply_rotary_emb``) resolve the op when the module is
    imported, which a subprocess does under the neuron backend; the result is diffusers'."""
    code = (
        "import torch\n"
        "from diffusers.models.embeddings import apply_rotary_emb as diffusers_rope\n"
        "import difflet.layers.embeddings as E\n"
        "from difflet.backends.neuron.ops_impl import embeddings as N\n"
        "assert E.apply_rotary_emb_fp32 is N.apply_rotary_emb_fp32, E.apply_rotary_emb_fp32\n"
        "gen = torch.Generator().manual_seed(0)\n"
        "x = torch.randn(1, 3, 7, 16, generator=gen).bfloat16()\n"
        "cos = torch.randn(7, 16, generator=gen).bfloat16()\n"
        "sin = torch.randn(7, 16, generator=gen).bfloat16()\n"
        "out = E.apply_rotary_emb(x, torch.stack([cos, sin], -1))\n"
        "expect = diffusers_rope(x, (cos, sin))\n"
        "assert out.dtype == torch.bfloat16 and torch.equal(out, expect)\n"
    )
    env = dict(os.environ, DIFFLET_BACKEND="neuron")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
