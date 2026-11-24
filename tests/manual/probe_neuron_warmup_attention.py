"""Where Wan's 413 s warm-up goes: the first call of the neuron attention op at long sequences.

Times difflet's neuron ``attention`` (torch-neuronx's flash-attention kernel for [B, H, S, d] bf16)
at Wan 2.2's per-rank shapes at TP4, called as Wan calls it ((B*H, S, d), tp_q/tp_k) (10 of 40 heads, head size 128), eager and under
``torch.compile(backend="neuron", fullgraph=True)`` as the backend compiles its blocks. Each case:
call 1 (first-call set-up + run), calls 2 and 3 (run only). One process; run it twice to see whether
anything persists between processes.

    NEURON_RT_VISIBLE_CORES=0 DIFFLET_BACKEND=neuron python probe_neuron_warmup_attention.py --tag a
Prints one JSON line per case.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

SHAPES = {  # name: (batch, heads, seq_q, seq_k, head_dim)
    "wan81_self": (1, 10, 32760, 32760, 128),
    "wan81_cross": (1, 10, 32760, 512, 128),
    "wan9_self": (1, 10, 4680, 4680, 128),
    # the 81-frame self-attention split into equal query chunks (every chunk attends all keys)
    "wan81_self_q2": (1, 10, 16380, 32760, 128),
    "wan81_self_q4": (1, 10, 8190, 32760, 128),
    "wan81_self_q8": (1, 10, 4095, 32760, 128),
}
#: calls of a chunk shape that make up one full 81-frame self-attention
CHUNKS = {"wan81_self_q2": 2, "wan81_self_q4": 4, "wan81_self_q8": 8}


def timed(fn, *args) -> float:
    start = time.perf_counter()
    out = fn(*args)
    out.cpu()  # forces execution and synchronizes
    return time.perf_counter() - start


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="a")
    p.add_argument("--shapes", default=",".join(SHAPES))
    p.add_argument("--modes", default="eager,compiled")
    args = p.parse_args()

    import torch_neuronx  # noqa: F401  registers the neuron device

    from difflet.backends.neuron.ops_impl.attention import attention

    device = torch.device("neuron")
    torch.manual_seed(0)
    def wan_attention(q, k, v):  # as Wan calls it: heads in the batch, the standard (S, D) layout
        return attention(q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), causal=False,
                         attention_mask=None, tp_q=True, tp_k=True, tp_out=False)

    for mode in args.modes.split(","):
        fn = wan_attention
        if mode == "compiled":
            fn = torch.compile(wan_attention, backend="neuron", dynamic=False, fullgraph=True)
        for name in args.shapes.split(","):
            b, h, sq, sk, d = SHAPES[name]
            q = torch.randn(b * h, sq, d, dtype=torch.bfloat16).to(device)
            k = torch.randn(b * h, sk, d, dtype=torch.bfloat16).to(device)
            v = torch.randn(b * h, sk, d, dtype=torch.bfloat16).to(device)
            calls = [round(timed(fn, q, k, v), 3) for _ in range(3)]
            n = CHUNKS.get(name, 1)
            print(json.dumps({"tag": args.tag, "mode": mode, "shape": name, "first_s": calls[0],
                              "second_s": calls[1], "third_s": calls[2],
                              "setup_s": round(calls[0] - calls[1], 3),
                              "full_attention_steady_s": round(n * calls[2], 3)}), flush=True)


if __name__ == "__main__":
    main()
