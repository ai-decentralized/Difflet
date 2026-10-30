"""Repro for AWS: eager bool bitwise_and on a neuron tensor with a non-contiguous operand is wrong.

    python tests/manual/repro_neuron_bool_bitwise_strided.py

m[j] = j < 50 over L = 256; M = m repeated over rows, so M & M^T should be the 50 x 50 block
(2,500 True). On the neuron device, with the transposed view as an operand, the result has
12,800 True (as if the operand were M^T alone). uint8 and int64 are correct; making the transposed
operand contiguous gives the right answer; a 64 x 64 random case was correct.
"""

import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401

L = 256
for dtype in (torch.bool, torch.uint8, torch.int64):
    m = torch.zeros(1, L, dtype=dtype)
    m[:, :50] = 1
    cpu = m.view(1, 1, 1, L).repeat(1, 1, L, 1)
    want = cpu & cpu.transpose(2, 3)
    dev = cpu.to("neuron")
    got = (dev & dev.transpose(2, 3)).cpu()
    fixed = (dev & dev.transpose(2, 3).contiguous()).cpu()
    print(f"{str(dtype):12s} strided operand: nonzero {int((got != 0).sum())} (want {int((want != 0).sum())}), "
          f"equal={torch.equal(got, want)}; contiguous operand: equal={torch.equal(fixed, want)}")
