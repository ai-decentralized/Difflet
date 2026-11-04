"""Layer 2 of the TPU profile: where does one Wan block's time go, today?

``benchmark/wan_block_profile.py`` answered this once, on torch_xla, before the
fused attention kernel: self-attention was 68% of a 20 ms block. Everything it
measured has moved since -- torch_tpu replaced torch_xla, and the step is now
``torch.compile``d (360 ms/step against 608) -- so this is the same ablation,
routed through ``device_runtime`` so it runs on either runtime, plus the pieces
a first hand-written kernel would target:

* RoPE, the two LayerNorm + adaLN modulations, the qk-norm (with and without
  its cross-rank reduce);
* a "skeleton" variant with both attentions and the FFN removed, which leaves
  exactly the element-wise work around them (norms, modulation, gates,
  residuals, the fp32 round trips).

Each variant removes one piece and re-times; the difference is that piece's
cost. Compilation can change fusion when a piece is removed, so a difference is
an estimate, not an exact share -- the trace (``profiler_probe.py``) is the
check.

    DIFFLET_TPU_COMPILE=1 \\
        python -m benchmark.tpu_profile.wan_block_ablation --out /mnt/models/profile/block.json

``DIFFLET_TPU_COMPILE=1`` compiles each variant afresh (``torch._dynamo.reset``
between them, since the variants swap submodules and module globals);
``DIFFLET_TPU_PALLAS=1`` puts the Pallas flash kernel under self-attention.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

R = 8
LAYERS = 40
S, T, D, F, HEADS, HD = 4680, 512, 5120, 13824, 40, 128
PEAK_FLOPS = 197e12


def layer_flops(world: int) -> float:
    """Useful FLOPs one rank does for one block (see wan_2_2.md for the count)."""
    linear = 2 * S * (6 * D * D + 2 * D * F)
    self_attn = 4 * HEADS * S * S * HD
    cross = 4 * HEADS * S * T * HD + 2 * 2 * D * D * T
    return (linear + self_attn + cross) / world


def worker(rank, world, trace_dir, reply_q):
    try:
        _worker(rank, world, trace_dir, reply_q)
    except Exception as exc:  # noqa: BLE001
        import traceback
        reply_q.put({"done": rank, "error": repr(exc), "traceback": traceback.format_exc()})
        raise


def _worker(rank, world, trace_dir, reply_q):
    from difflet.backends.tpu.ops_impl import device_runtime

    device_runtime.initialize_replica(rank, world)

    import torch
    import torch.nn.functional as F_

    torch.set_num_threads(max(1, (os.cpu_count() or world) // world))

    from difflet.backends.tpu.ops_impl import parallel_mesh
    from difflet.backends.tpu.ops_impl.platform import configure_matmul_precision
    from difflet.models.wan import modeling_wan as M
    from difflet.pipeline.parallel_mesh import MeshSpec

    configure_matmul_precision()
    parallel_mesh.init_parallel_mesh(MeshSpec(tp=world))
    from difflet.backends.tpu.ops_impl import collectives

    collectives.warm_process_groups()  # before any compile (DIFFLET_TPU_FUNCOL)

    device = device_runtime.device()
    dtype = torch.bfloat16
    compiled = os.environ.get(device_runtime.COMPILE_ENV, "").strip().lower() in ("1", "true", "yes")

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
    flops = layer_flops(world)

    def report(label, per, mfu_flops=None):
        if rank != 0:
            return
        row = {"label": label, "per_layer_ms": per * 1000, "step_estimate_s": per * LAYERS}
        if mfu_flops:
            row["mfu"] = mfu_flops / per / PEAK_FLOPS
        reply_q.put(row)

    def prepare(fn):
        if not compiled:
            return fn
        torch._dynamo.reset()
        return torch.compile(fn, backend="tpu")

    def timed(label, fn, mfu_flops=None):
        fn = prepare(fn)
        with torch.no_grad():
            x = h
            for _ in range(2):           # warm (and, compiled, trace) the graph
                x = fn(x)
            device_runtime.wait()
            x = h
            mark = time.monotonic()
            for _ in range(R):
                x = fn(x)
            device_runtime.wait()
        per = (time.monotonic() - mark) / R
        report(label, per, mfu_flops)
        return per

    full = lambda x: block(x, ctx, temb, rope)  # noqa: E731
    timed("full block", full, flops)

    # Layer 3: an op-level device trace of the full block. jax.profiler is the
    # tracer that sees torch_tpu's device ops (profiler_probe.py, 2026-10-27).
    if trace_dir and rank == 0:
        import jax

        jax.profiler.start_trace(trace_dir)
        try:
            timed("full block (traced)", full, flops)
        finally:
            jax.profiler.stop_trace()
    elif trace_dir:
        timed("full block (traced)", full, flops)  # keep the collectives in step

    # --- ablations, each removing exactly one piece -----------------------

    class _PassThrough(torch.nn.Module):
        """A correctly-shaped zero-cost stand-in for a sub-block."""

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    def swapped(label, **parts):
        real = {name: getattr(block, name) for name in parts}
        for name, stand_in in parts.items():
            setattr(block, name, stand_in)
        try:
            timed(label, full)
        finally:
            for name, module in real.items():
                setattr(block, name, module)

    def patched(label, owner, name, stand_in):
        real = getattr(owner, name)
        setattr(owner, name, stand_in)
        try:
            timed(label, full)
        finally:
            setattr(owner, name, real)

    swapped("no self-attn", attn1=_PassThrough())
    swapped("no cross-attn", attn2=_PassThrough())
    swapped("no ffn", ffn=_PassThrough())
    # What is left is the element-wise work: norms, modulation, gates, residuals.
    swapped("skeleton (no attn1, attn2, ffn)",
            attn1=_PassThrough(), attn2=_PassThrough(), ffn=_PassThrough())
    swapped("no LayerNorms (norm1/2/3)",
            norm1=torch.nn.Identity(), norm2=torch.nn.Identity(), norm3=torch.nn.Identity())

    # The stand-in must carry the QUERY's length: cross-attention's v is 512 long.
    patched("attention kernel -> passthrough", M, "_attn_kernel",
            lambda q, k, v, *, head_dim: q)
    patched("no RoPE", M, "apply_rotary_emb", lambda x, cos, sin: x)

    def _local_rms_norm(self, norm, x):
        local_sq = x.float().pow(2).sum(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(local_sq / x.shape[-1] + 1e-6)).to(x.dtype)

    patched("qk-norm without the cross-rank reduce", M.WanAttention, "_global_rms_norm",
            _local_rms_norm)
    patched("no qk-norm", M.WanAttention, "_global_rms_norm", lambda self, norm, x: x)

    # Row-parallel all-reduces: 3 per layer, (1, S, D) bf16 each. The layers
    # bind the TPU reduce when they are built, so turn each one's reduce off.
    from difflet.backends.tpu.ops_impl.linear import RowParallelLinear

    rows = [m for m in block.modules() if isinstance(m, RowParallelLinear)]
    for m in rows:
        m.reduce_output = False
    try:
        timed("no row-parallel all-reduce", full)
    finally:
        for m in rows:
            m.reduce_output = True

    # --- isolated primitives ---------------------------------------------
    local = D // world
    heads_local = HEADS // world

    def bare(label, fn, shape, mfu_flops=None):
        fn = prepare(fn)
        with torch.no_grad():
            acc = torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(2):
                acc = acc + fn()
            device_runtime.wait()
            mark = time.monotonic()
            for _ in range(R):
                acc = acc + fn()
            device_runtime.wait()
        report(label, (time.monotonic() - mark) / R, mfu_flops)

    q = torch.randn(1, heads_local, S, HD, dtype=dtype, device=device)
    k = torch.randn(1, heads_local, S, HD, dtype=dtype, device=device)
    v = torch.randn(1, heads_local, S, HD, dtype=dtype, device=device)
    attn_flops = 4 * heads_local * S * S * HD
    bare("  [bare] sdpa self 10x4680x4680",
         lambda: F_.scaled_dot_product_attention(q, k, v), (1, heads_local, S, HD), attn_flops)
    bare("  [bare] difflet attention op (flash if enabled)",
         lambda: M._attn_kernel(q, k, v, head_dim=HD), (1, heads_local, S, HD), attn_flops)

    big = torch.randn(1, S, D, dtype=dtype, device=device)
    bare("  [bare] all-reduce (1,4680,5120)", lambda: C.reduce_tp(big), (1, S, D))
    # Bandwidth curve: is 48 MB in ~0.84 ms near the ICI limit, or is a bigger
    # message faster per byte? Ring bus bandwidth = 2 (n-1)/n * bytes / time.
    for rows in (S // 4, S // 2, 2 * S):
        msg = torch.randn(1, rows, D, dtype=dtype, device=device)
        bare(f"  [bare] all-reduce (1,{rows},5120) {rows * D * 2 / 2**20:.0f} MiB",
             lambda msg=msg: C.reduce_tp(msg), (1, rows, D))

    w = torch.randn(D, local, dtype=dtype, device=device)
    bare("  [bare] matmul S x 5120 x 1280", lambda: big @ w, (1, S, local), 2 * S * D * local)
    wf = torch.randn(D, F // world, dtype=dtype, device=device)
    bare("  [bare] matmul S x 5120 x 3456", lambda: big @ wf, (1, S, F // world),
         2 * S * D * (F // world))

    shift = torch.randn(1, 1, D, dtype=torch.float32, device=device)
    scale = torch.randn(1, 1, D, dtype=torch.float32, device=device)
    bare("  [bare] LayerNorm + adaLN modulate (1,4680,5120)",
         lambda: (block.norm1(big.float()) * (1 + scale) + shift).type_as(big), (1, S, D))

    if rank == 0:
        reply_q.put({"label": "mem_gb",
                     "per_layer_ms": device_runtime.memory_info(device)["bytes_used"] / 2**30,
                     "step_estimate_s": 0})
    reply_q.put({"done": rank})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-dir", help="also write a jax.profiler trace of the full block (rank 0)")
    args = parser.parse_args()

    from difflet.backends.tpu.ops_impl import device_runtime

    world = len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    device_runtime.prepare_launch(world)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, args.trace_dir, reply_q)) for r in range(world)]
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
            mfu = f"  MFU {msg['mfu']:.0%}" if "mfu" in msg else ""
            print(f"{msg['label']:48s} {msg['per_layer_ms']:8.2f} ms/layer "
                  f"-> {msg['step_estimate_s']:6.2f} s/step{mfu}", flush=True)
    for p in procs:
        p.join(timeout=60)
    meta = {"runtime": "torch_tpu",
            "compile": os.environ.get(device_runtime.COMPILE_ENV, ""),
            "pallas": os.environ.get("DIFFLET_TPU_PALLAS", ""), "world": world}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
