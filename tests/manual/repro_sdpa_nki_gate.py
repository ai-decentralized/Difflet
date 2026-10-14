"""Standalone repro: on the neuron device, F.scaled_dot_product_attention reaches torch-neuronx's
NKI flash-attention kernel only when the query and key lengths are multiples of 512.

No Difflet code: torch and torch-neuronx only. Single process, one NeuronCore:

    python tests/manual/repro_sdpa_nki_gate.py [--json result.json] [--no-compiled]

Per case (bf16, batch 1, 40 heads of 128, unmasked, non-causal, scale 1/sqrt(128)):

* gate: what torch-neuronx's ``_can_use_nki_flash_attention`` returned for the call (recorded by
  wrapping it), in eager mode and while ``torch.compile(backend="neuron")`` traced the call;
* eager SDPA, compiled SDPA, and the NKI kernel called directly
  (``scaled_dot_product_attention_kernel``): median ms over 10 synchronised calls after warm-up;
* parity of the SDPA and direct-kernel outputs against a CPU fp32 reference (max abs error).

The 4,680-token cases are Wan 2.1/2.2 at 480x832x9 (latent 3x60x104, patch 1x2x2): self-attention
4,680 x 4,680 and cross-attention to the 512 text tokens.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from importlib import metadata

import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401  registers the neuron device
from torch_neuronx.neuron_dynamo_backend import decompositions
from torch_neuronx.python_ops import scaled_dot_product_fused_attention as fused_sdpa
from torch_neuronx.python_ops.nki_kernels.scaled_dot_product_attention import (
    scaled_dot_product_attention_kernel,
)

HEADS, HEAD_DIM = 40, 128
CASES = (
    ("Wan 480x832x9 self-attention", 4680, 4680),
    ("Wan 480x832x9 cross-attention", 4680, 512),
    ("4096 (8 x 512)", 4096, 4096),
    ("4608 (9 x 512)", 4608, 4608),
    ("5120 (10 x 512)", 5120, 5120),
)

decisions: list[bool] = []
_gate = decompositions._can_use_nki_flash_attention


def _recorded_gate(*args, **kwargs):
    result = _gate(*args, **kwargs)
    decisions.append(bool(result))
    return result


decompositions._can_use_nki_flash_attention = _recorded_gate
if hasattr(fused_sdpa, "_can_use_nki_flash_attention"):
    fused_sdpa._can_use_nki_flash_attention = _recorded_gate


def median_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.neuron.synchronize()
        samples = []
        for _ in range(iters):
            torch.neuron.synchronize()
            start = time.perf_counter()
            fn()
            torch.neuron.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def gate_for(fn) -> bool | None:
    decisions.clear()
    with torch.no_grad():
        fn()
    torch.neuron.synchronize()
    return decisions[-1] if decisions else None


def max_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (out.float().cpu() - ref).abs().max().item()


def run_case(label: str, lq: int, lk: int, compiled: bool) -> dict:
    generator = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, HEADS, n, HEAD_DIM, generator=generator) for n in (lq, lk, lk))
    reference = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    qd, kd, vd = (t.to(torch.bfloat16).to("neuron") for t in (q, k, v))
    scale = HEAD_DIM ** -0.5

    def sdpa():
        return F.scaled_dot_product_attention(qd, kd, vd)

    def kernel():
        return scaled_dot_product_attention_kernel(qd, kd, vd, scale=scale, training=False)

    row = {"case": label, "lq": lq, "lk": lk,
           "lq_multiple_of_512": lq % 512 == 0, "lk_multiple_of_512": lk % 512 == 0}
    row["gate_eager"] = gate_for(sdpa)
    row["sdpa_eager_ms"] = median_ms(sdpa)
    with torch.no_grad():
        row["sdpa_max_err"] = max_err(sdpa(), reference)
    if compiled:
        compiled_sdpa = torch.compile(sdpa, backend="neuron", dynamic=False, fullgraph=True)
        row["gate_compiled"] = gate_for(compiled_sdpa)
        row["sdpa_compiled_ms"] = median_ms(compiled_sdpa)
    row["kernel_ms"] = median_ms(kernel)
    with torch.no_grad():
        row["kernel_max_err"] = max_err(kernel(), reference)
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", default=None)
    p.add_argument("--no-compiled", action="store_true", help="skip the torch.compile column")
    args = p.parse_args()

    build = {name: metadata.version(name) for name in ("torch", "torch-neuronx", "nki", "neuronx-cc")}
    rows = [run_case(label, lq, lk, not args.no_compiled) for label, lq, lk in CASES]

    header = (f"{'case':32s} {'Lq':>5s} {'Lk':>5s} {'gate eager':>10s} {'gate comp':>9s} "
              f"{'SDPA eager':>10s} {'SDPA comp':>9s} {'kernel':>8s} {'err SDPA':>9s} {'err kern':>9s}")
    print(header)
    for r in rows:
        print(f"{r['case']:32s} {r['lq']:5d} {r['lk']:5d} {str(r['gate_eager']):>10s} "
              f"{str(r.get('gate_compiled', '-')):>9s} {r['sdpa_eager_ms']:8.2f}ms "
              f"{r.get('sdpa_compiled_ms', float('nan')):7.2f}ms {r['kernel_ms']:6.2f}ms "
              f"{r['sdpa_max_err']:9.2e} {r['kernel_max_err']:9.2e}", flush=True)
    print(f"build: {build}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"build": build, "heads": HEADS, "head_dim": HEAD_DIM, "dtype": "bfloat16",
                       "rows": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
