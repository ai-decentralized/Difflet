#!/usr/bin/env python3
"""0c: what does TorchTPU's matmul precision cost, and what does it buy?

`ops_impl/platform.py` pins XLA's matmul precision to "highest" because the TPU
default measured ~1e-2 absolute error per matmul under torch_xla, which made the
>= 0.9995 cosine parity target unreachable. TorchTPU defaults to `medium` (bf16
MXU, fp32 accumulate) and prices `high` at ~3x and `highest` at ~6x MXU cost.

Measured on FLUX's own block geometry, four deep, against a CPU fp32 oracle.

**Cosine is reported as 1-cos.** A first run of this spike printed 1.000000 for
every mode: at a relative error of 1e-3, cosine over a 12M-element tensor is
1 - 5e-7, which rounds to 1 at six decimals. Cosine has no resolution here, so
max relative error is the metric that decides and cosine is kept only because
0.9995 is the bar the repo quotes.

Rotary embeddings are skipped deliberately — they are elementwise, and matmul
precision cannot touch them.
"""


import contextlib
import time

import torch
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock

DIM, HEADS, HEAD_DIM = 3072, 24, 128     # FLUX.1-dev's real DiT geometry
IMG_TOKENS, TXT_TOKENS = 4096, 512       # 1024^2 at patch 2, plus the T5 stream
N_BLOCKS = 4
ITERS = 3

TARGET_COS = 0.9995


def build_blocks() -> torch.nn.ModuleList:
    torch.manual_seed(0)
    return torch.nn.ModuleList(
        FluxTransformerBlock(dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM)
        for _ in range(N_BLOCKS)
    ).eval()


def make_inputs():
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(1, IMG_TOKENS, DIM, generator=g),
        torch.randn(1, TXT_TOKENS, DIM, generator=g),
        torch.randn(1, DIM, generator=g),
    )


@torch.no_grad()
def forward_capture(blocks, h, e, t):
    outs = []
    for blk in blocks:
        e, h = blk(hidden_states=h, encoder_hidden_states=e, temb=t)
        outs.append(h)
    return outs


def errors(got: torch.Tensor, ref: torch.Tensor):
    got, ref = got.flatten().double(), ref.flatten().double()
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
    abs_err = (got - ref).abs()
    denom = ref.abs().clamp_min(1e-6)
    return 1.0 - cos, abs_err.max().item(), (abs_err / denom).max().item()


@torch.no_grad()
def timed(fn) -> float:
    fn()
    torch.accelerator.synchronize()
    started = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - started) / ITERS


@contextlib.contextmanager
def attention_only_highest(blocks):
    """Raise precision inside each block's attention, leave the rest on medium.

    The first version of this spike wrapped the whole forward in the context,
    which is just the global setting by another name — and measured identically
    to it. Scoping has to be selective to mean anything.
    """
    originals = [blk.attn.forward for blk in blocks]
    for blk, original in zip(blocks, originals):
        def wrapped(*args, _orig=original, **kwargs):
            with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
                return _orig(*args, **kwargs)
        blk.attn.forward = wrapped
    try:
        yield
    finally:
        for blk, original in zip(blocks, originals):
            blk.attn.forward = original


def report(label, outs, oracle, seconds):
    rows = [errors(o.float().cpu(), r) for o, r in zip(outs, oracle)]
    one_minus_cos, max_abs, max_rel = rows[-1]
    ok = (1.0 - one_minus_cos) >= TARGET_COS
    ms = seconds * 1000 / N_BLOCKS
    print(
        f"{label:<36} {'PASS' if ok else 'FAIL'}  "
        f"1-cos={one_minus_cos:9.3e}  max|abs|={max_abs:9.3e}  max|rel|={max_rel:9.3e}  "
        f"{ms:6.1f} ms/block"
    )
    print(f"{'':38}depth 1-cos: {' '.join(f'{r[0]:.2e}' for r in rows)}")
    return (1.0 - one_minus_cos), ms


def main():
    print(f"FLUX block dim={DIM} heads={HEADS} head_dim={HEAD_DIM}, "
          f"{IMG_TOKENS}+{TXT_TOKENS} tokens, {N_BLOCKS} blocks, bar cos >= {TARGET_COS}\n")

    blocks = build_blocks()
    h, e, t = make_inputs()
    print("building CPU fp32 oracle ...", flush=True)
    oracle = [o.float() for o in forward_capture(blocks, h, e, t)]
    print("oracle done\n", flush=True)

    dev = torch.device("tpu")
    blocks = blocks.to(dev)
    h, e, t = h.to(dev), e.to(dev), t.to(dev)
    run = lambda: forward_capture(blocks, h, e, t)  # noqa: E731

    results = {}
    for mode in ("medium", "high", "highest"):
        torch.set_float32_matmul_precision(mode)
        results[mode] = report(f"global float32_matmul={mode}", run(), oracle, timed(run))

    torch.set_float32_matmul_precision("medium")
    try:
        with attention_only_highest(blocks):
            results["attn-only highest"] = report(
                "medium + attention-only HIGHEST", run(), oracle, timed(run)
            )
    except Exception as exc:  # noqa: BLE001
        print(f"attention-only scoping -> {type(exc).__name__}: {exc}")

    print()
    passing = {m: v for m, v in results.items() if v[0] >= TARGET_COS}
    if passing:
        best = min(passing, key=lambda m: passing[m][1])
        base = results["highest"][1]
        print(f"VERDICT: cheapest setting at cos >= {TARGET_COS} is '{best}' at "
              f"{passing[best][1]:.1f} ms/block — {base / passing[best][1]:.2f}x "
              f"faster than today's global highest ({base:.1f} ms/block)")
    else:
        print(f"VERDICT: no setting reaches cos >= {TARGET_COS}")


if __name__ == "__main__":
    main()
