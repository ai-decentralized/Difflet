"""Where does Wan's 899 ms/step go? Ablate one real transformer block.

One WanTransformerBlock maps (B, S, D) -> (B, S, D), so R blocks can be
chained with a mark_step per iteration — exactly the shape of the real
workload, amortising launch overhead over R without letting XLA fuse the
iterations into one graph.

Each variant swaps out one piece and re-times. The differences are the
per-piece cost; absolute per-variant numbers are per *layer*, so x40 for a
step.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

R = 8
S, T, D, F, HEADS, HD = 4680, 512, 5120, 13824, 40, 128


def worker(rank, world, reply_q):
    try:
        _worker(rank, world, reply_q)
    except Exception as exc:  # noqa: BLE001
        import traceback
        reply_q.put({"done": rank, "error": repr(exc),
                     "traceback": traceback.format_exc()})
        raise


def _worker(rank, world, reply_q):
    from torch_xla._internal import pjrt

    pjrt.initialize_multiprocess(rank, world)

    import torch
    import torch.nn.functional as F_
    import torch_xla
    import torch_xla.core.xla_model as xm

    torch.set_num_threads(max(1, (os.cpu_count() or world) // world))

    from difflet.backends.tpu.ops_impl import parallel_mesh
    from difflet.backends.tpu.ops_impl.platform import configure_matmul_precision
    from difflet.models.wan import modeling_wan as M
    from difflet.pipeline.parallel_mesh import MeshSpec

    configure_matmul_precision()
    parallel_mesh.init_parallel_mesh(MeshSpec(tp=world))

    device = torch_xla.device()
    dtype = torch.bfloat16

    block = M.WanTransformerBlock(
        dim=D, ffn_dim=F, num_heads=HEADS, cross_attn_norm=True, eps=1e-6, dtype=dtype
    )
    rank_util = M.SPMDRank(world)
    object.__setattr__(block.attn1, "_rank_util", rank_util)
    object.__setattr__(block.attn2, "_rank_util", rank_util)
    block.requires_grad_(False)
    block = block.eval().to(device)

    h = torch.randn(1, S, D, dtype=dtype, device=device)
    ctx = torch.randn(1, T, D, dtype=dtype, device=device)
    temb = torch.randn(1, 6, D, dtype=dtype, device=device)
    cos = torch.randn(1, S, 1, HD, dtype=torch.float32, device=device)
    sin = torch.randn(1, S, 1, HD, dtype=torch.float32, device=device)
    rope = (cos, sin)

    def timed(label, fn):
        with torch.no_grad():
            x = h
            for _ in range(2):           # warm the graph
                x = fn(x)
                xm.mark_step()
            xm.wait_device_ops()
            x = h
            mark = time.monotonic()
            for _ in range(R):
                x = fn(x)
                xm.mark_step()
            xm.wait_device_ops()
        per = (time.monotonic() - mark) / R
        if rank == 0:
            reply_q.put({"label": label, "per_layer_ms": per * 1000,
                         "step_estimate_s": per * 40})
        return per

    full = lambda x: block(x, ctx, temb, rope)  # noqa: E731
    timed("full block", full)

    # --- ablations, each removing exactly one piece -----------------------
    identity = torch.nn.Identity()

    real_attn1, real_attn2, real_ffn = block.attn1, block.attn2, block.ffn

    class _PassThrough(torch.nn.Module):
        """Return a correctly-shaped zero-cost stand-in for a sub-block."""

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    block.attn1 = _PassThrough()
    timed("no self-attn", full)
    block.attn1 = real_attn1

    block.attn2 = _PassThrough()
    timed("no cross-attn", full)
    block.attn2 = real_attn2

    block.ffn = _PassThrough()
    timed("no ffn", full)
    block.ffn = real_ffn

    # SDPA itself, keeping every projection and collective in place.
    real_kernel = M._attn_kernel
    # the stand-in must carry the QUERY's sequence length: cross-attention's
    # v is 512 long while the block's residual expects 4680.
    M._attn_kernel = lambda q, k, v, *, head_dim: q
    timed("attn matmuls -> passthrough", full)
    M._attn_kernel = real_kernel

    # The cross-rank qk-norm: 4 extra all-reduces per layer (q,k x attn1,attn2).
    real_norm = M.WanAttention._global_rms_norm

    def _local_rms_norm(self, norm, x):
        local_sq = x.float().pow(2).sum(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(local_sq / x.shape[-1] + 1e-6)
        return x_normed.to(x.dtype)

    M.WanAttention._global_rms_norm = _local_rms_norm
    timed("qk-norm without the cross-rank reduce", full)
    M.WanAttention._global_rms_norm = real_norm

    # Row-parallel all-reduces: 2 per layer, (1, S, D) bf16 each.
    from difflet.backends.tpu.ops_impl import collectives as C

    real_reduce = C.reduce_tp
    C.reduce_tp = lambda x: x
    import difflet.backends.tpu.ops_impl.linear as Lin
    real_lin_reduce = Lin.reduce_tp
    Lin.reduce_tp = lambda x: x
    timed("no row-parallel all-reduce", full)
    C.reduce_tp = real_reduce
    Lin.reduce_tp = real_lin_reduce

    # --- isolated primitives ---------------------------------------------
    local = D // world
    heads_local = HEADS // world

    def bare(label, fn, shape):
        with torch.no_grad():
            acc = torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(2):
                acc = acc + fn()
                xm.mark_step()
            xm.wait_device_ops()
            mark = time.monotonic()
            for _ in range(R):
                acc = acc + fn()
                xm.mark_step()
            xm.wait_device_ops()
        per = (time.monotonic() - mark) / R
        if rank == 0:
            reply_q.put({"label": label, "per_layer_ms": per * 1000,
                         "step_estimate_s": per * 40})

    q = torch.randn(1, heads_local, S, HD, dtype=dtype, device=device)
    k = torch.randn(1, heads_local, S, HD, dtype=dtype, device=device)
    v = torch.randn(1, heads_local, S, HD, dtype=dtype, device=device)
    bare("  [bare] sdpa self 10x4680x4680",
         lambda: F_.scaled_dot_product_attention(q, k, v), (1, heads_local, S, HD))

    big = torch.randn(1, S, D, dtype=dtype, device=device)
    bare("  [bare] all-reduce (1,4680,5120)", lambda: C.reduce_tp(big), (1, S, D))

    w = torch.randn(D, local, dtype=dtype, device=device)
    bare("  [bare] matmul S x 5120 x 1280", lambda: big @ w, (1, S, local))

    wf = torch.randn(D, F // world, dtype=dtype, device=device)
    bare("  [bare] matmul S x 5120 x 3456", lambda: big @ wf, (1, S, F // world))

    if rank == 0:
        reply_q.put({"label": "mem",
                     "per_layer_ms": xm.get_memory_info(device)["bytes_used"] / 2**30,
                     "step_estimate_s": 0})
    reply_q.put({"done": rank})


def main() -> None:
    world = len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, reply_q)) for r in range(world)]
    for p in procs:
        p.start()
    seen, rows = 0, []
    while seen < world:
        if not any(p.is_alive() for p in procs) and reply_q.empty():
            print("all workers exited early", flush=True)
            break
        try:
            msg = reply_q.get(timeout=120)
        except Exception:  # noqa: BLE001
            continue
        if "done" in msg:
            seen += 1
            if msg.get("traceback"):
                print(msg["traceback"], flush=True)
        else:
            rows.append(msg)
            print(f"{msg['label']:44s} {msg['per_layer_ms']:8.2f} ms/layer "
                  f"-> {msg['step_estimate_s']:6.2f} s/step", flush=True)
    for p in procs:
        p.join(timeout=60)
    Path("/mnt/models/wan_block_profile.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
