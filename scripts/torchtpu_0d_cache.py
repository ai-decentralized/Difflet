#!/usr/bin/env python3
"""0d: does a compiled TPU executable survive a fresh process?

This is the spike the whole migration rests on. `core/application_base.py`
records that torch_xla cannot do it — `UNIMPLEMENTED: Deserializing serialized
executable not supported` — which is why every fresh v5e worker pays XLA's
compile again (44-170 s of DiT compile) and why warm start is 268.8 s on LTX-2
against trn2's 58.5 s.

TorchTPU claims a tier-2 (machine-local, cross-process) and tier-3 (filesystem,
cross host and restart) cache holding serialized executables. The knobs are
TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT and TORCH_TPU_TIER2_COMPILATION_CACHE —
found in the shipped binary; they are in neither the docs nor the public API
list.

Run as one process per condition; the caller compares. Timing starts at import,
so t_first_forward isolates compile from the CPU-side build.
"""

import json
import os
import sys
import time

T0 = time.perf_counter()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from diffusers.models.transformers.transformer_flux import (  # noqa: E402
    FluxTransformerBlock,
)

DIM, HEADS, HEAD_DIM = 3072, 24, 128
IMG_TOKENS, TXT_TOKENS = 4096, 512
N_BLOCKS = 4


@torch.no_grad()
def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    rank = 0
    if "RANK" in os.environ:
        # No set_device_index here. The launcher already gives each worker its
        # own chip, so every process sees exactly one addressable device and
        # set_device_index(local_rank) raises "Device index N is out of bounds
        # of the number of addressable devices 1" for every rank but 0 — which
        # deadlocks rank 0 in the rendezvous waiting for peers that just died.
        dist.init_process_group(backend="tpu_dist")
        rank = dist.get_rank()
    t_import = time.perf_counter() - T0

    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(
        FluxTransformerBlock(dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM)
        for _ in range(N_BLOCKS)
    ).eval()
    g = torch.Generator().manual_seed(1)
    h = torch.randn(1, IMG_TOKENS, DIM, generator=g)
    e = torch.randn(1, TXT_TOKENS, DIM, generator=g)
    t = torch.randn(1, DIM, generator=g)

    dev = torch.device("tpu")
    started = time.perf_counter()
    blocks, h, e, t = blocks.to(dev), h.to(dev), e.to(dev), t.to(dev)
    torch.accelerator.synchronize()
    t_to_device = time.perf_counter() - started

    def forward():
        hh, ee = h, e
        for blk in blocks:
            ee, hh = blk(hidden_states=hh, encoder_hidden_states=ee, temb=t)
        torch.accelerator.synchronize()

    started = time.perf_counter()
    forward()
    t_first = time.perf_counter() - started

    started = time.perf_counter()
    forward()
    t_second = time.perf_counter() - started

    print("0D_RESULT " + json.dumps({
        "label": label,
        "rank": rank,
        "tier3_root": os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", ""),
        "t_import_s": round(t_import, 2),
        "t_to_device_s": round(t_to_device, 2),
        "t_first_forward_s": round(t_first, 2),
        "t_second_forward_s": round(t_second, 3),
    }), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
