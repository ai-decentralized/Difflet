"""Manual check: the parity runner (tests/numerical/neuron_parity.py) on hardware.

Run on a trn2 host with the TorchNeuron stack (single process), from the repository root:

    PYTHONPATH=. python tests/manual/check_neuron_parity_c12.py [results.json]

Reruns the C2, C3 and C9 cases through ``run_parity``, judged by the TPU numerical gate
(``check``), and a saved device output that must reload and compare exactly. The optional
argument writes the results and the build as JSON.
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ["DIFFLET_BACKEND"] = "neuron"

from difflet.backends.neuron.compile import configure_compile_cache

configure_compile_cache()  # before torch: importing torch autoloads torch_neuronx

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from difflet import ops  # noqa: E402
from difflet.backends.neuron.compile import compile_blocks  # noqa: E402
from tests.numerical import neuron_parity as parity  # noqa: E402

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
        parity.run_parity("RMSNorm", rms, (x_norm,), dtype=torch.float32),
        parity.run_parity("RMSNorm", rms, (x_norm,), dtype=torch.bfloat16),
        parity.run_parity("attention 4680 (SDPA)", attention, (q, k, v), dtype=torch.float32),
        parity.run_parity("attention 4680 (NKI kernel)", attention, (q, k, v), dtype=torch.bfloat16),
        parity.run_parity("toy 12 blocks, eager", toy, (x_toy,), dtype=torch.bfloat16),
        parity.run_parity("toy 12 blocks, compiled", toy, (x_toy,), dtype=torch.bfloat16,
                          prepare=compile_blocks),
    ]

    with torch.no_grad():
        model = Toy(depth=2).to(dtype=torch.bfloat16, device="neuron").eval()
        x_dev = x_toy.to(dtype=torch.bfloat16, device="neuron")
        first, second = model(x_dev).cpu(), model(x_dev).cpu()
    with tempfile.TemporaryDirectory() as tmp:
        parity.save_reference(os.path.join(tmp, "toy"), {"out": first}, {"note": "roundtrip"})
        saved, _ = parity.load_reference(os.path.join(tmp, "toy"))
    roundtrip = parity.metrics(second, saved["out"])
    ok = roundtrip["max_abs"] == 0.0
    results.append({"name": "saved reference roundtrip", "dtype": "bfloat16", "metrics": roundtrip,
                    "control": None, "fallbacks": [], "ok": ok, "reasons": [] if ok else ["not exact"],
                    "notes": []})
    return parity.report(results, json_path=sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    sys.exit(main())
