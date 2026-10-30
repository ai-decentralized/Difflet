"""attention_cte's per-query key bounds, through the neuron attention op, on one core.

    python tests/manual/probe_neuron_flash_bounds.py

Cases ([B*H, L, 128] bf16, non-causal): key padding (every query: keys [0, valid)), at L = 4,096
and at L = 27,456 (HunyuanVideo 1.5, 61 frames, per Ulysses rank: 25,440 image + 1,985 text +
31 more); per-row varying bounds at L = 4,096. Each against an fp32 CPU reference computed in
query chunks (no L x L tensor), plus the time per call. The long case is the one the dense-mask
path cannot run.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401

from difflet.backends.neuron.ops_impl import attention as A  # noqa: E402

D = 128


def reference(q, k, v, lo, hi, scale, chunk=1024):
    """fp32 attention with keys [lo_i, hi_i) per query row, in query chunks."""
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1])
    keys = torch.arange(k.shape[1])
    for s in range(0, q.shape[1], chunk):
        e = min(s + chunk, q.shape[1])
        scores = (q[:, s:e].float() @ k.float().transpose(1, 2)) * scale
        allowed = (keys.view(1, 1, -1) >= lo[:, s:e]) & (keys.view(1, 1, -1) < hi[:, s:e])
        scores = scores.masked_fill(~allowed, float("-inf"))
        out[:, s:e] = torch.softmax(scores, -1) @ v.float()
    return out


def run(name, bh, length, lo, hi):
    g = torch.Generator().manual_seed(length)
    q, k, v = (torch.randn(bh, length, D, generator=g).to(torch.bfloat16) for _ in range(3))
    scale = D ** -0.5
    dev = [t.to("neuron") for t in (q, k, v)]
    print(f"{name}: bounded kernel available={A._FLASH_BOUNDED_OP is not None}, "
          f"flash-eligible={A._can_use_flash_kernel(*dev, None)}", flush=True)
    times = []
    for _ in range(3):
        t = time.perf_counter()
        with torch.no_grad():
            out = A.attention(*dev, scale=scale, causal=False, bound_min=lo, bound_max=hi,
                              tp_q=True, tp_k=True, tp_out=False)
        torch.neuron.synchronize()
        times.append(time.perf_counter() - t)
    out = out.float().cpu()
    want = reference(q, k, v, lo, hi, scale)
    err = ((out - want).abs().mean() / want.abs().mean()).item()
    print(f"  finite={bool(torch.isfinite(out).all())} rel vs fp32 {err:.2e}; "
          f"{times[0] * 1e3:.0f} ms first, {min(times[1:]) * 1e3:.1f} ms after", flush=True)


def key_padding(bh, length, valid):
    lo = torch.zeros(bh, length, 1, dtype=torch.int32)
    return lo, torch.full((bh, length, 1), valid, dtype=torch.int32)


run("key padding, 4 x 4096, 3,800 valid keys", 4, 4096, *key_padding(4, 4096, 3800))
g = torch.Generator().manual_seed(7)
lo = torch.randint(0, 2048, (4, 4096, 1), generator=g, dtype=torch.int32)
hi = lo + torch.randint(1, 2048, (4, 4096, 1), generator=g, dtype=torch.int32)
run("per-row bounds, 4 x 4096", 4, 4096, lo, hi)
run("key padding, 4 x 27456 (HV1.5 61 frames per CP rank)", 4, 27456, *key_padding(4, 27456, 25440 + 120))
