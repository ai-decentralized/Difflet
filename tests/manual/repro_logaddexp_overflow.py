"""Repro for AWS: torch.logaddexp compiled on the neuron backend is wrong for inputs above ~80.

    python tests/manual/repro_logaddexp_overflow.py

Eager PyTorch computes logaddexp stably (max + log1p(exp(-|a - b|))); compiled with
backend="neuron" the result is garbage near 80 and inf from ~90, consistent with a naive
log(exp(a) + exp(b)). Attention logsumexps reach this range, so merging partial attentions
(ring attention) with logaddexp breaks under torch.compile.
"""

import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401

a = torch.tensor([10.0, 80.0, 90.0, 100.0, 300.0])
b = a - 0.5
fn = lambda x, y: torch.logaddexp(x, y)  # noqa: E731
compiled = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
print("inputs  :", a.tolist(), "and each minus 0.5")
print("eager   :", fn(a, b).tolist())
print("compiled:", compiled(a.to("neuron:0"), b.to("neuron:0")).cpu().tolist())
