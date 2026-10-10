"""Manual check: neuron backend per-block compilation on hardware.

Run on a trn2 host with the TorchNeuron stack (single process), twice: the first
run compiles on a cold cache, the second shows the warm start. Caches live under
``<DIFFLET_COMPILE_CACHE>/_neuron``; remove that directory for a cold run.

    python tests/manual/check_neuron_compile_c9.py

A model-free toy with 12 identical pre-norm transformer blocks (hidden 1024,
8 heads of 128, 1,000 tokens, bf16) uses difflet.ops attention, so the NKI flash
kernel runs inside the compiled blocks at a length that is not a multiple of 512.
Twelve blocks exceed Dynamo's default per-code-object cache of 8 entries, so
reuse of one graph across blocks is required, not incidental.

Pass criteria: no graph break (fullgraph=True); parameter names unchanged;
compiled output no further from CPU fp32 than 1.5x eager's error; at most 2
unique graphs for 12 blocks and none added by a second call; identical second
call; no CPU fallbacks.
"""

from __future__ import annotations

import os
import sys
import time

os.environ["DIFFLET_BACKEND"] = "neuron"

from difflet.backends.neuron.cache import apply_neuron_cache_env

CACHE_ENV = apply_neuron_cache_env()  # before torch: importing torch autoloads torch_neuronx

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402

from difflet.backends.neuron.compile import compile_blocks  # noqa: E402
from difflet.backends.neuron.runtime import track_fallbacks  # noqa: E402
from difflet.ops import attention  # noqa: E402

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
        a = attention(heads(q), heads(k), heads(v), scale=HEAD_DIM**-0.5, tp_q=True, tp_k=True, tp_out=False)
        x = x + self.out(a.view(b, HEADS, s, HEAD_DIM).transpose(1, 2).reshape(b, s, D))
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x)), approximate="tanh"))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def forward(model, x):
    with torch.no_grad():
        out = model(x)
    torch.neuron.synchronize()
    return out.to("cpu", torch.float64)


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} {detail}", flush=True)
    return ok


def main() -> int:
    torch.manual_seed(0)
    model = Toy().eval()
    x = torch.randn(1, S, D)
    with torch.no_grad():
        ref = model(x).double()
    model = model.to(dtype=torch.bfloat16, device="neuron")
    x_dev = x.to(dtype=torch.bfloat16, device="neuron")
    eager = forward(model, x_dev)

    results = []
    names_before = list(model.state_dict())
    stacks = compile_blocks(model)
    results.append(report("stacks found, names unchanged", stacks == ["blocks"] and list(model.state_dict()) == names_before, f"stacks={stacks}"))

    counters.clear()
    try:
        with track_fallbacks() as fallbacks:
            t0 = time.perf_counter()
            first = forward(model, x_dev)
            first_s = time.perf_counter() - t0
            graphs_first = counters["stats"]["unique_graphs"]
            second = forward(model, x_dev)
    except Exception as exc:  # fullgraph=True turns a graph break into an error
        report("compiled forward", False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        return 1
    graphs_second = counters["stats"]["unique_graphs"]

    eager_err = (eager - ref).abs().mean().item()
    compiled_err = (first - ref).abs().mean().item()
    results.append(report("parity against CPU fp32", compiled_err <= 1.5 * eager_err + 1e-6,
                          f"mean_err compiled={compiled_err:.2e} eager={eager_err:.2e}"))
    results.append(report("one graph shared by the blocks", graphs_first <= 2 and graphs_second == graphs_first,
                          f"unique graphs: {graphs_first} for {DEPTH} blocks, {graphs_second} after 2nd call"))
    results.append(report("second call identical", torch.equal(first, second)))
    results.append(report("no CPU fallbacks", not fallbacks, f"{fallbacks}"))
    print(f"first compiled forward: {first_s:.1f} s (cold cache: compile; warm cache: load)", flush=True)
    print(f"caches: NEFF {CACHE_ENV['TORCH_NEURONX_NEFF_CACHE_DIR']}, HLO {CACHE_ENV['TORCH_NEURONX_HLO_CACHE_DIR']}", flush=True)
    print(f"{sum(results)}/{len(results)} passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
