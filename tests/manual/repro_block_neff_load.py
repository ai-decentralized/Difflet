"""Standalone repro: N identical torch.compile'd blocks share one compiled program (NEFF); is that
program loaded onto the NeuronCore once, or once per block?

No Difflet code: torch and torch-neuronx only. Single process, one NeuronCore:

    python tests/manual/repro_block_neff_load.py --cache-dir /tmp/blk --label cold   # compiles
    python tests/manual/repro_block_neff_load.py --cache-dir /tmp/blk --label warm   # same cache

The model is a stack of ``--blocks`` identical pre-norm MLP blocks (LayerNorm, Linear H->4H,
GELU, Linear 4H->H, residual), bf16, each compiled with
``block.compile(backend="neuron", fullgraph=True, dynamic=False)``: same code, same shapes,
weights as graph inputs, so Dynamo produces one graph and the cache one NEFF for all blocks.
The first forward runs the blocks one at a time with a device sync after each, timing every
block's first call; a second forward is timed whole. On the warm run nothing compiles
(CompilationCache.TotalCompilations stays 0), so a block's first-call time is tracing plus
loading its program: a slow block 0 and fast blocks 1..N-1 mean the program is loaded once; every
block slow means it is loaded per call site.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="NEFF/HLO cache root (empty for the cold run, reused for the warm run)")
    p.add_argument("--label", default="run")
    p.add_argument("--blocks", type=int, default=16)
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Before torch: importing torch autoloads torch_neuronx, which reads these once.
    root = args.cache_dir.expanduser().resolve()
    for var, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"),
                     ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "neff_local"),
                     ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo")):
        os.environ[var] = str(root / sub)
        (root / sub).mkdir(parents=True, exist_ok=True)

    import torch
    import torch_neuronx  # noqa: F401
    import torch_neuronx.metrics as metrics

    metrics.set_enabled(True)

    def counters() -> dict:
        names = ("TotalCompilations", "PersistentHits", "InMemoryHits")
        return {n: metrics.get_counter_value("CompilationCache." + n) or 0 for n in names}

    class Block(torch.nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.norm = torch.nn.LayerNorm(hidden)
            self.up = torch.nn.Linear(hidden, 4 * hidden)
            self.down = torch.nn.Linear(4 * hidden, hidden)

        def forward(self, x):
            return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))

    torch.manual_seed(0)
    start = time.perf_counter()
    blocks = torch.nn.ModuleList(Block(args.hidden) for _ in range(args.blocks))
    blocks = blocks.to(device="neuron", dtype=torch.bfloat16).eval()
    for block in blocks:
        block.compile(backend="neuron", fullgraph=True, dynamic=False)
    x = torch.randn(1, args.tokens, args.hidden, dtype=torch.bfloat16).to("neuron")
    torch.neuron.synchronize()
    build_s = time.perf_counter() - start

    before = counters()
    first_call_s = []
    with torch.no_grad():
        h = x
        for block in blocks:
            t = time.perf_counter()
            h = block(h)
            torch.neuron.synchronize()
            first_call_s.append(time.perf_counter() - t)
        after_first = counters()
        t = time.perf_counter()
        h2 = x
        for block in blocks:
            h2 = block(h2)
        torch.neuron.synchronize()
        second_forward_s = time.perf_counter() - t

    neffs = sorted(root.glob("neff/**/*.neff"))
    result = {
        "label": args.label, "blocks": args.blocks, "tokens": args.tokens, "hidden": args.hidden,
        "build_s": build_s, "first_call_s": first_call_s,
        "first_forward_s": sum(first_call_s), "second_forward_s": second_forward_s,
        "counters_before": before, "counters_after_first_forward": after_first,
        "neff_files": len(neffs), "largest_neff_bytes": max((n.stat().st_size for n in neffs), default=0),
        "versions": {"torch": torch.__version__, "torch_neuronx": torch_neuronx.__version__},
    }
    print(f"[{args.label}] blocks={args.blocks} tokens={args.tokens} hidden={args.hidden} "
          f"build={build_s:.1f}s second_forward={second_forward_s:.3f}s "
          f"compilations={after_first['TotalCompilations'] - before['TotalCompilations']} "
          f"neffs={len(neffs)} largest_neff={result['largest_neff_bytes'] / 1e6:.1f} MB", flush=True)
    print(f"[{args.label}] first call per block (s): "
          + " ".join(f"{s:.2f}" for s in first_call_s), flush=True)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
