"""Manual check: neuron backend attention (NKI flash kernel, SDPA fallback) on hardware.

Run on a trn2 host with the TorchNeuron stack (single process):

    python tests/manual/check_neuron_attention_c3.py

At Wan 2.2 A14B shapes (40 heads of 128, 4,680 video tokens, 512 text tokens),
through difflet.ops with DIFFLET_BACKEND=neuron:

* parity: fp32 on the device (SDPA, exact) against CPU fp32 (1e-3 relative); bf16 on
  the device (the NKI kernel when unmasked) no further from CPU fp32 than 1.5x CPU
  bf16; a boolean key-padding mask as well (SDPA in both dtypes);
* zero CPU fallbacks;
* eager time per call against the CPU backend's explicit matmul-softmax on the
  device. Unmasked calls take the NKI flash kernel directly, so a large speedup
  is expected; no speedup means they fell back to the decomposed SDPA path.
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time

os.environ["DIFFLET_BACKEND"] = "neuron"

import torch

from difflet import ops
from difflet.backends.cpu.ops_impl.attention import attention as matmul_softmax_attention
from difflet.backends.neuron.runtime import track_fallbacks

HEADS, HEAD_DIM, S_VIDEO, S_TEXT = 40, 128, 4680, 512
SCALE = 1.0 / math.sqrt(HEAD_DIM)
FLAGS = dict(scale=SCALE, causal=False, tp_q=True, tp_k=True, tp_out=False)


def call(fn, q, k, v, device, dtype, **extra):
    args = [t.to(device=device, dtype=dtype) for t in (q, k, v)]
    extra = {name: (t.to(device) if torch.is_tensor(t) else t) for name, t in extra.items()}
    with torch.no_grad():
        out = fn(*args, **FLAGS, **extra)
    return out.to("cpu", torch.float64)


def check(name, q, k, v, **extra):
    ref = call(ops.attention, q, k, v, "cpu", torch.float32, **extra)
    cpu_bf16 = call(ops.attention, q, k, v, "cpu", torch.bfloat16, **extra)
    with track_fallbacks() as fallbacks:
        dev_fp32 = call(ops.attention, q, k, v, "neuron", torch.float32, **extra)
        dev_bf16 = call(ops.attention, q, k, v, "neuron", torch.bfloat16, **extra)
    fp32_err = (dev_fp32 - ref).abs().max().item() / max(1.0, ref.abs().max().item())
    cpu_err = (cpu_bf16 - ref).abs().mean().item()
    dev_err = (dev_bf16 - ref).abs().mean().item()
    ok = fp32_err <= 1e-3 and dev_err <= 1.5 * cpu_err + 1e-6 and not fallbacks
    print(f"{'PASS' if ok else 'FAIL'}  {name:30s} fp32 rel_err={fp32_err:.1e}  bf16 mean_err "
          f"device={dev_err:.2e} cpu={cpu_err:.2e}  fallbacks={fallbacks}", flush=True)
    return ok


def time_ms(fn, q, k, v, warmup=3, iters=10):
    args = [t.to(device="neuron", dtype=torch.bfloat16) for t in (q, k, v)]
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args, **FLAGS)
        torch.neuron.synchronize()
        times = []
        for _ in range(iters):
            torch.neuron.synchronize()
            t0 = time.perf_counter()
            fn(*args, **FLAGS)
            torch.neuron.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def main() -> int:
    torch.manual_seed(0)
    q = torch.randn(HEADS, S_VIDEO, HEAD_DIM)
    k_self, v_self = torch.randn(HEADS, S_VIDEO, HEAD_DIM), torch.randn(HEADS, S_VIDEO, HEAD_DIM)
    k_text, v_text = torch.randn(HEADS, S_TEXT, HEAD_DIM), torch.randn(HEADS, S_TEXT, HEAD_DIM)
    keep = torch.ones(HEADS, S_VIDEO, S_TEXT, dtype=torch.bool)
    keep[..., 40:] = False  # 40 real text tokens, the rest padding

    results = [
        check("self-attention 4680x4680", q, k_self, v_self),
        check("cross-attention 4680x512", q, k_text, v_text),
        check("cross-attention, padding mask", q, k_text, v_text, attention_mask=keep),
    ]

    sdpa = time_ms(ops.attention, q, k_self, v_self)
    explicit = time_ms(matmul_softmax_attention, q, k_self, v_self)
    print(f"self-attention bf16, eager: ops.attention {sdpa:.2f} ms, matmul-softmax {explicit:.2f} ms, "
          f"speedup {explicit / sdpa:.1f}x", flush=True)

    print(f"{sum(results)}/{len(results)} parity checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
