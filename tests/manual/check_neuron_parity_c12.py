"""Manual check: the parity harness reproduces the earlier hardware results.

Run on a trn2 host with the TorchNeuron stack (single process):

    python tests/manual/check_neuron_parity_c12.py [results.json]

Expected: every check passes, matching the C2, C3 and C9 manual checks, and a
saved device output reloads and compares exactly. The optional argument writes
the results and the build as JSON.
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ["DIFFLET_BACKEND"] = "neuron"

from difflet.backends.neuron.cache import apply_neuron_cache_env

apply_neuron_cache_env()  # before torch: importing torch autoloads torch_neuronx

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from difflet import ops  # noqa: E402
from difflet.backends.neuron import parity  # noqa: E402
from difflet.backends.neuron.compile import compile_blocks  # noqa: E402

D, HEADS, S = 1024, 8, 1000
HEAD_DIM = D // HEADS


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.out = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.ff1, self.ff2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

    def forward(self, x):
        b, s, _ = x.shape

        def heads(t):
            return t.view(b, s, HEADS, HEAD_DIM).transpose(1, 2).reshape(b * HEADS, s, HEAD_DIM)

        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        a = ops.attention(heads(q), heads(k), heads(v), scale=HEAD_DIM**-0.5, tp_q=True, tp_k=True, tp_out=False)
        x = x + self.out(a.view(b, HEADS, s, HEAD_DIM).transpose(1, 2).reshape(b, s, D))
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x)), approximate="tanh"))


class Toy(nn.Module):
    def __init__(self, depth=12):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def attention(q, k, v):
    return ops.attention(q, k, v, scale=128**-0.5, tp_q=True, tp_k=True, tp_out=False)


def main() -> int:
    torch.manual_seed(0)
    rms = ops.RMSNorm(5120, eps=1e-6)
    rms.weight.data = 1 + 0.1 * torch.randn(5120)
    x_norm = torch.randn(1, 4680, 5120)
    q, k, v = (torch.randn(40, 4680, 128) for _ in range(3))
    toy, x_toy = Toy(), torch.randn(1, S, D)

    results = [
        parity.check("RMSNorm", rms, (x_norm,), dtype=torch.float32),
        parity.check("RMSNorm", rms, (x_norm,), dtype=torch.bfloat16),
        parity.check("attention 4680 (SDPA)", attention, (q, k, v), dtype=torch.float32, fp32_rtol=1e-3),
        parity.check("attention 4680 (NKI kernel)", attention, (q, k, v), dtype=torch.bfloat16),
        parity.check("toy 12 blocks, eager", toy, (x_toy,), dtype=torch.bfloat16),
        parity.check("toy 12 blocks, compiled", toy, (x_toy,), dtype=torch.bfloat16, prepare=compile_blocks),
    ]

    with torch.no_grad():
        model = Toy(depth=2).to(dtype=torch.bfloat16, device="neuron").eval()
        x_dev = x_toy.to(dtype=torch.bfloat16, device="neuron")
        first, second = model(x_dev).cpu(), model(x_dev).cpu()
    with tempfile.TemporaryDirectory() as tmp:
        parity.save_reference(os.path.join(tmp, "toy"), {"out": first}, {"note": "C12 roundtrip"})
        saved, _ = parity.load_reference(os.path.join(tmp, "toy"))
    results.append(parity.compare_to_reference("saved reference roundtrip", second, saved["out"]))

    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(parity.results_json(results))
    return parity.report(results)


if __name__ == "__main__":
    sys.exit(main())
