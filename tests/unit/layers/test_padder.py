"""Unit tests for difflet.layers.padder (pure shape math, no backend)."""

import torch

from difflet.layers.padder import (
    MaybePadder,
    pad,
    pad_interleaved,
    pad_sizes,
    round_up_to_divisor,
)


def test_round_up_to_divisor():
    assert round_up_to_divisor(0, 8) == 0
    assert round_up_to_divisor(1, 8) == 8
    assert round_up_to_divisor(8, 8) == 8
    assert round_up_to_divisor(9, 8) == 16
    assert round_up_to_divisor(15, 5) == 15


def test_pad_sizes_single_dim_right():
    # shape (3,) padded on dim 0 to size 5 -> right pad 2
    out = pad_sizes((3,), 0, 5, left=False)
    assert out == (0, 2)


def test_pad_sizes_single_dim_left():
    out = pad_sizes((3,), 0, 5, left=True)
    assert out == (2, 0)


def test_pad_sizes_no_padding_returns_none():
    # already big enough -> no padding -> None
    assert pad_sizes((5,), 0, 5) is None
    # truncation is not performed; still no padding
    assert pad_sizes((7,), 0, 5) is None


def test_pad_sizes_multi_dim_interleaving():
    # 3-D shape, pad dim 0 and dim 2; F.pad order is reversed (last dim first)
    out = pad_sizes((2, 3, 4), (0, 2), (4, 6), left=False)
    # dim2 needs 2, dim0 needs 2; reversed lhs=[0,0,0] rhs=[2,0,2]
    # reversed(rhs)=[2,0,2]; zipped with reversed(lhs)=[0,0,0]
    assert out == (0, 2, 0, 0, 0, 2)


def test_pad_sizes_int_dims_and_sizes_broadcast():
    out = pad_sizes((2, 2), (0, 1), 4, left=False)
    assert out == (0, 2, 0, 2)


def test_pad_tensor_none_passthrough():
    assert pad(None, 0, 4) is None


def test_pad_tensor_applies_padding():
    t = torch.ones(3)
    out = pad(t, 0, 5)
    assert out.shape == (5,)
    assert torch.equal(out, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))


def test_pad_tensor_no_padding_returns_same():
    t = torch.ones(5)
    out = pad(t, 0, 5)
    assert out is t


def test_pad_parameter_is_detached():
    p = torch.nn.Parameter(torch.ones(3))
    out = pad(p, 0, 5)
    assert not out.requires_grad
    assert out.shape == (5,)


def test_pad_interleaved_basic():
    t = torch.tensor([1.0, 2.0, 3.0])
    out = pad_interleaved(t, dim=0, size=9, source_len_per_group=1, pad_len_per_group=2)
    assert torch.equal(out, torch.tensor([1.0, 0, 0, 2.0, 0, 0, 3.0, 0, 0]))


def test_pad_interleaved_2d():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = pad_interleaved(t, dim=0, size=4, source_len_per_group=1, pad_len_per_group=1)
    expected = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    assert torch.equal(out, expected)


def test_maybe_padder_end_mode():
    padder = MaybePadder(size=5, padding="end")
    t = torch.ones(3)
    out = padder(t, dim=0)
    assert out.shape == (5,)


def test_maybe_padder_end_mode_none_passthrough():
    padder = MaybePadder(size=5, padding="end")
    assert padder(None, dim=0) is None


def test_maybe_padder_interleaved_none_passthrough():
    padder = MaybePadder(size=6, padding="interleaved", interleaved_factor=1)
    assert padder(None, dim=0) is None


def test_maybe_padder_interleaved_simple():
    # interleaved_factor groups: source per group = shape // factor
    padder = MaybePadder(size=6, padding="interleaved", interleaved_factor=3)
    t = torch.tensor([1.0, 2.0, 3.0])
    out = padder(t, dim=0)
    # new_size=6, source_len_per_group = 3//3 = 1, pad_len = (6-3)//3 = 1
    assert torch.equal(out, torch.tensor([1.0, 0, 2.0, 0, 3.0, 0]))


def test_maybe_padder_interleaved_with_split_size():
    # KV-style: reshape into (split_size, ?) then interleave on split dim
    padder = MaybePadder(
        size=8, padding="interleaved", split_size=2, interleaved_factor=2
    )
    # weight shape (4,) split into (2, 2); pad split dim 2->4 interleaved
    t = torch.arange(4, dtype=torch.float32)
    out = padder(t, dim=0)
    assert out.shape == (8,)
