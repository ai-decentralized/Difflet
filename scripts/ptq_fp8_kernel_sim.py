#!/usr/bin/env python3
"""CPU-simulate the NKI FP8 linear kernel (nki.simulate) on small shapes and compare with the CPU
reference of the same fp8 math. Fast iteration on kernel tracing / numerics without the device.

    NKI_FP8_E4M3_MODE=non_ocp python scripts/ptq_fp8_kernel_sim.py
"""
from __future__ import annotations

import math
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NKI_FP8_E4M3_MODE", "non_ocp")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402
import torch  # noqa: E402

from difflet.backends.trainium.nki_kernels.fp8_linear import (  # noqa: E402
    fp8_linear_static_kernel, fp8_linear_token_kernel)

FP8_MAX = 240.0


def metrics(ref, test):
    ref = ref.float().flatten(); test = test.float().flatten()
    err = (test - ref).pow(2).sum().item(); sig = ref.pow(2).sum().item()
    cos = torch.nn.functional.cosine_similarity(ref, test, dim=0).item()
    return round(cos, 6), round(10 * math.log10(sig / err), 2) if err > 0 else float("inf")


def case(S, K, N, mode, with_bias=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(S, K, generator=g) * (1 + 3 * (torch.rand(S, 1, generator=g) > 0.9))).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g) / math.sqrt(K)).to(torch.bfloat16)
    bias = (torch.randn(1, N, generator=g) * 0.01).to(torch.bfloat16)
    s_w = (w.float().abs().max() / FP8_MAX).reshape(1, 1)
    w8 = (w.float() / s_w).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    s_in = (x.float().abs().max() * 1.25 / FP8_MAX).reshape(1, 1)
    if mode == "static":
        q = (x.float() / s_in).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s_in
    else:
        st = (x.float().abs().amax(-1, keepdim=True) / FP8_MAX * (1 + 2 ** -7)).clamp_min(1 / (240 * 512))
        q = (x.float() / st).to(torch.float8_e4m3fn).float() * st
    ref = q @ (w8.float() * s_w).t() + (bias.float() if with_bias else 0)
    kernel = fp8_linear_static_kernel if mode == "static" else fp8_linear_token_kernel
    y = nki.simulate(kernel)(x, w8.t().contiguous(), s_w.float(), s_in.float(), bias if with_bias else None)
    y = torch.as_tensor(y)
    return metrics(ref, y)


def main():
    ok = True
    for S, K, N in ((128, 256, 512), (200, 384, 640), (130, 512, 1024)):
        for mode in ("static", "token"):
            try:
                cos, snr = case(S, K, N, mode)
                print(f"[sim] S={S} K={K} N={N} {mode:6s} cos {cos} snr {snr} dB", flush=True)
                ok &= cos > 0.9999
            except Exception:
                ok = False
                print(f"[sim] S={S} K={K} N={N} {mode} FAILED\n{traceback.format_exc()[-2500:]}", flush=True)
    print("SIM_OK" if ok else "SIM_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
