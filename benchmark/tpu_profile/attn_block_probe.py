"""FLUX's attention shape on one chip: torch_tpu's SDPA (what FLUX runs today, the
``scaled_dot_product_cudnn_attention`` custom call) against the JAX Pallas flash
kernel (``DIFFLET_TPU_PALLAS=1``'s path) over a sweep of block sizes.

    python -m benchmark.tpu_profile.attn_block_probe [--heads 6 --seq 4608]

Each side runs in its own process (torch_tpu and a bare JAX client would both
claim the chip). Per call: median of 5 timed runs of 50 back-to-back calls,
after warm-up; prints us/call, TFLOP/s (4*H*S^2*D) and the max relative error
against an fp32 reference of the same bf16 inputs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

ITERS, REPEATS = 50, 5


def _inputs(heads, seq, dim):
    import numpy as np

    rng = np.random.default_rng(0)
    return [rng.standard_normal((1, heads, seq, dim)).astype(np.float32) for _ in range(3)]


def _reference(q, k, v):
    import numpy as np

    scale = 1.0 / np.sqrt(q.shape[-1])
    s = np.einsum("bhqd,bhkd->bhqk", q, k) * scale
    s = np.exp(s - s.max(-1, keepdims=True))
    return np.einsum("bhqk,bhkd->bhqd", s / s.sum(-1, keepdims=True), v)


def _torch_side(args, reply_q):
    try:
        from difflet.backends.tpu.ops_impl import device_runtime

        device_runtime.initialize_replica(0, 1)
        import numpy as np
        import torch
        import torch.nn.functional as F

        dev = device_runtime.device()
        q, k, v = (torch.from_numpy(a).to(torch.bfloat16).to(dev)
                   for a in _inputs(args.heads, args.seq, args.dim))
        scale = 1.0 / args.dim ** 0.5

        @torch.compile(backend="tpu")
        def many(q, k, v):
            out = q
            for _ in range(ITERS):  # chained so the calls cannot be dropped
                out = F.scaled_dot_product_attention(out, k, v, scale=scale)
            return out

        @torch.compile(backend="tpu")
        def once(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, scale=scale)

        got = once(q, k, v).float().cpu().numpy()
        many(q, k, v).float().cpu()
        times = []
        for _ in range(REPEATS):
            mark = time.perf_counter()
            many(q, k, v)[0, 0, 0, 0].item()
            times.append((time.perf_counter() - mark) / ITERS)
        reply_q.put({"impl": "torch_tpu sdpa", "us": float(np.median(times)) * 1e6, "out": got})
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        reply_q.put({"impl": "torch_tpu sdpa", "error": repr(exc)})


def _jax_side(args, reply_q):
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.experimental.pallas.ops.tpu import flash_attention as fa

        q, k, v = (jnp.asarray(a, dtype=jnp.bfloat16) for a in _inputs(args.heads, args.seq, args.dim))
        scale = 1.0 / args.dim ** 0.5
        rows = []
        for spec in args.blocks.split(","):
            bq, bkm, bk = (int(x) for x in spec.split("/"))
            sizes = fa.BlockSizes(block_q=bq, block_k_major=bkm, block_k=bk, block_b=1)

            def call(q, k, v, sizes=sizes):
                return fa.flash_attention(q * scale, k, v, sm_scale=1.0, block_sizes=sizes)

            def many(q, k, v, call=call):
                return jax.lax.fori_loop(0, ITERS, lambda _, o: call(o, k, v), q)

            try:
                once_j, many_j = jax.jit(call), jax.jit(many)
                got = np.asarray(once_j(q, k, v).astype(jnp.float32))
                many_j(q, k, v).block_until_ready()
                times = []
                for _ in range(REPEATS):
                    mark = time.perf_counter()
                    many_j(q, k, v).block_until_ready()
                    times.append((time.perf_counter() - mark) / ITERS)
                rows.append({"impl": f"pallas flash {spec}", "us": float(np.median(times)) * 1e6,
                             "out": got})
            except Exception as exc:  # noqa: BLE001 -- a block size the kernel rejects
                rows.append({"impl": f"pallas flash {spec}", "error": repr(exc)[:200]})
        reply_q.put(rows)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        reply_q.put([{"impl": "pallas flash", "error": repr(exc)}])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--seq", type=int, default=4608)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--blocks", default="512/512/512,1024/512/512,1024/1024/512,"
                                            "1024/1024/1024,2048/1024/512,512/1024/1024")
    args = parser.parse_args()
    import numpy as np

    from difflet.backends.tpu.ops_impl import device_runtime

    ref = _reference(*_inputs(args.heads, args.seq, args.dim))
    flops = 4.0 * args.heads * args.seq * args.seq * args.dim
    ctx = mp.get_context("spawn")
    rows = []
    device_runtime.prepare_launch(1)
    for target in (_torch_side, _jax_side):
        reply_q = ctx.Queue()
        proc = ctx.Process(target=target, args=(args, reply_q))
        proc.start()
        got = reply_q.get(timeout=1800)
        proc.join(timeout=120)
        rows += got if isinstance(got, list) else [got]
    print(f"shape [1,{args.heads},{args.seq},{args.dim}], {flops / 1e9:.1f} GFLOP per call")
    for row in rows:
        if "error" in row:
            print(f"  {row['impl']:28s} ERROR {row['error']}")
            continue
        err = float(np.abs(row["out"] - ref).max() / np.abs(ref).max())
        print(f"  {row['impl']:28s} {row['us']:8.1f} us  {flops / row['us'] / 1e6:6.1f} TFLOP/s"
              f"  max rel err {err:.2e}")
    print(json.dumps({"done": True}))


if __name__ == "__main__":
    main()
