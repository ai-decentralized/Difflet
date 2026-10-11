"""Manual check: the benchmark harness gives stable timings on hardware.

Run on a trn2 host with the TorchNeuron stack (single process), from the repository root:

    PYTHONPATH=. python tests/manual/check_neuron_benchmark_c13.py [results.json]

Times a model-free toy (12 identical transformer blocks, hidden 1024, 8 heads of
128, 1,000 tokens, bf16) in eager and compiled modes, interleaved over rounds,
with the allocator cache released between cases. Pass: neither case is flagged
unstable or drifting. The optional argument writes every sample, the peak memory
and the build as JSON.
"""

from __future__ import annotations

import os
import sys

os.environ["DIFFLET_BACKEND"] = "neuron"

from difflet.backends.neuron.compile import configure_compile_cache

configure_compile_cache()  # before torch: importing torch autoloads torch_neuronx

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from difflet import ops  # noqa: E402
from difflet.backends.neuron.compile import compile_blocks  # noqa: E402
from tests.numerical import neuron_benchmark as benchmark  # noqa: E402

DEPTH, D, HEADS, S = 12, 1024, 8, 1000
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
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def main() -> int:
    torch.manual_seed(0)
    eager = Toy().to(dtype=torch.bfloat16, device="neuron").eval()
    compiled = Toy().to(dtype=torch.bfloat16, device="neuron").eval()
    compiled.load_state_dict(eager.state_dict())
    compile_blocks(compiled)
    x = torch.randn(1, S, D).to(dtype=torch.bfloat16, device="neuron")

    def forward(model):
        def call():
            with torch.no_grad():
                model(x)
        return call

    results = benchmark.run(
        {"toy 12 blocks, eager": forward(eager), "toy 12 blocks, compiled": forward(compiled)},
        warmup=3, iters=10, rounds=5,
        release_cache=benchmark.neuron_release_cache,
        memory_stats=benchmark.neuron_memory_stats,
    )
    record = benchmark.report(results, {"depth": DEPTH, "hidden": D, "heads": HEADS, "tokens": S, "dtype": "bfloat16"})
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(record)
    stable = all(not r.flags for r in results)
    print(f"{'PASS' if stable else 'FAIL'}  timings stable (spread <= {benchmark.SPREAD_LIMIT:.0%}, "
          f"drift <= {benchmark.DRIFT_LIMIT})", flush=True)
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
