"""MiniMax-H3's Omni Transformer on a v5e-4: bf16 vs int8 weight-only, one process per chip.

    python -m benchmark.tpu_profile.minimax_h3_int8_probe [--height 256 --width 448 \
        --num-frames 124 --text-tokens 256 --iters 3]

Feasibility probe, not a port. Per rank: build the diffusers-style DiT with the
AdaLN branch precomputed (``precomputed_adaln=True``: the 13B ``adaln_proj``
never reaches the device), TP-shard it like the Trainium port (q/k/v and the
SwiGLU up/gate column-parallel, ``to_out`` / down row-parallel, 14 heads per
chip), load this rank's slice of the real checkpoint, and run one denoise
forward at a fixed packed T2VA layout. Then, in the same process and on the same
inputs, quantize every block linear to int8 (symmetric, per output channel;
the scale is applied to the matmul output, which is the same as dequantizing
the weight) and run again.

Reports per variant: HBM in use, the first-call time (compile) and the warm
forward latency, and for int8 the relative L2 error and cosine of the video and
audio velocities against bf16.

What this does NOT check: the text conditioning is random (no Qwen3-VL), so
the error is that of one forward on synthetic text rows, not of a 30-step
video; the modulation table is the real one for t=0.5 (both slots). No VAE,
no scheduler. Weights stay int8 in HBM but the matmul runs in bf16 (v5e's int8
MXU path would need activation quantization too).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

MODEL = os.environ.get(
    "DIFFLET_MINIMAX_H3_PATH", "/mnt/models/hf/hub/models--MiniMaxAI--MiniMax-H3/snapshots"
)
SEED = 1234


def _transformer_path() -> str:
    path = Path(MODEL)
    if (path / "transformer").is_dir():
        return str(path / "transformer")
    return str(next(p for p in sorted(path.iterdir()) if (p / "transformer").is_dir()) / "transformer")


# ----------------------------------------------------------------------------- TP sharding

def _shard_for_tp(model, tp: int) -> None:
    """The Trainium port's split (backends/trainium/minimax_h3/transformer.py), TPU ops."""
    import torch.nn.functional as F
    from torch import nn

    from difflet.ops import ColumnParallelLinear, RowParallelLinear

    class SwiGLU(nn.Module):
        def __init__(self, hidden, ffn):
            super().__init__()
            self.up_proj = ColumnParallelLinear(hidden, ffn, bias=False, gather_output=False)
            self.gate_proj = ColumnParallelLinear(hidden, ffn, bias=False, gather_output=False)
            self.down_proj = RowParallelLinear(ffn, hidden, bias=False, input_is_parallel=True)

        def forward(self, x):
            return self.down_proj(self.up_proj(x) * F.silu(self.gate_proj(x)))

    def col(linear, gather):
        return ColumnParallelLinear(linear.in_features, linear.out_features,
                                    bias=linear.bias is not None, gather_output=gather)

    def row(linear):
        return RowParallelLinear(linear.in_features, linear.out_features,
                                 bias=linear.bias is not None, input_is_parallel=True)

    hidden, ffn = int(model.config.hidden_size), int(model.config.ffn_dim)
    model.context_embedder = col(model.context_embedder, True)
    model.norm_out.linear = col(model.norm_out.linear, True)
    blocks = list(model.token_refiner.refiner_blocks) + list(model.transformer_blocks)
    for block in blocks:
        attn = block.attn
        attn.heads = int(attn.heads) // tp
        attn.to_q, attn.to_k, attn.to_v = (col(attn.to_q, False), col(attn.to_k, False),
                                           col(attn.to_v, False))
        attn.to_out[0] = row(attn.to_out[0])
        attn.processor = _AttnProcessor()
        block.ff = SwiGLU(hidden, ffn)


class _AttnProcessor:
    """The stock processor with difflet's TPU attention op (flash when it applies)."""

    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        import math

        from difflet.models.minimax_h3.modeling_minimax_h3 import _apply_rotary_emb
        from difflet.ops import attention

        q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            q = _apply_rotary_emb(q, *rotary_emb)
            k = _apply_rotary_emb(k, *rotary_emb)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        b, h, s, d = q.shape
        out = attention(q.reshape(b * h, s, d), k.reshape(b * h, s, d), v.reshape(b * h, s, d),
                        scale=1.0 / math.sqrt(d), causal=False)
        out = out.reshape(b, h, s, d).transpose(1, 2).flatten(2, 3).to(q.dtype)
        return attn.to_out[1](attn.to_out[0](out.contiguous()))


def _rename(ffn: int):
    from difflet.backends.tpu.core.checkpoint import CheckpointSlice

    def rename(name: str):
        if ".ff.up_proj." in name:
            return CheckpointSlice(name.replace(".ff.up_proj.", ".ff.net.0.proj."), dim=0, start=0, stop=ffn)
        if ".ff.gate_proj." in name:
            return CheckpointSlice(name.replace(".ff.gate_proj.", ".ff.net.0.proj."), dim=0, start=ffn)
        if ".ff.down_proj." in name:
            return name.replace(".ff.down_proj.", ".ff.net.2.")
        return name

    return rename


# ----------------------------------------------------------------------------- int8

def quantize_block_linears_(model, group: int = 0) -> int:
    """Every Column/RowParallelLinear in the blocks -> int8 weight + bf16 scales.

    ``group=0``: one scale per output row, applied to the matmul output.
    ``group=g``: one scale per row and per ``g`` input columns; the weight is
    dequantized before the matmul (the scale no longer factors out of the sum).
    """
    import types

    import torch
    import torch.nn.functional as F

    from difflet.ops import ColumnParallelLinear, RowParallelLinear

    def _matmul(self, x):
        if self.w_scale.dim() == 1:
            return F.linear(x, self.weight_q.to(x.dtype)) * self.w_scale
        rows, cols = self.weight_q.shape
        w = self.weight_q.view(rows, -1, group).to(x.dtype) * self.w_scale.unsqueeze(-1)
        return F.linear(x, w.view(rows, cols))

    def col_forward(self, x):
        out = _matmul(self, x)
        if self.bias is not None:
            out = out + self.bias
        if self.gather_output:
            from difflet.ops import gather_tp_dim
            out = gather_tp_dim(out, dim=-1)
        return out

    def row_forward(self, x):
        from difflet.ops import reduce_tp

        out = _matmul(self, x)
        out = reduce_tp(out) if self.reduce_output else out
        return out if self.bias is None else out + self.bias

    count = 0
    blocks = list(model.token_refiner.refiner_blocks) + list(model.transformer_blocks)
    for block in blocks:
        for mod in block.modules():
            if not isinstance(mod, (ColumnParallelLinear, RowParallelLinear)):
                continue
            with torch.no_grad():
                w = mod.weight.data.float()
                rows, cols = w.shape
                if group:
                    w = w.view(rows, cols // group, group)
                scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
                q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8).view(rows, cols)
            dtype = mod.weight.dtype
            del mod.weight
            mod.register_buffer("weight_q", q)
            mod.register_buffer("w_scale", scale.squeeze(-1).to(dtype))
            is_row = isinstance(mod, RowParallelLinear)
            mod.forward = types.MethodType(row_forward if is_row else col_forward, mod)
            count += 1
    return count


# ----------------------------------------------------------------------------- inputs

def _inputs(args, cfg, transformer_path, mod_cache):
    import torch

    from difflet.models.minimax_h3.contracts import build_t2va_layout
    from difflet.models.minimax_h3.pipeline import build_adaln_modulation_table

    layout = build_t2va_layout(num_text_tokens=args.text_tokens, height=args.height,
                               width=args.width, num_frames=args.num_frames)
    g = torch.Generator().manual_seed(SEED)
    patch = int(cfg.in_channels) * 4
    video = torch.randn(1, len(layout.video_indices), patch, generator=g)
    audio = torch.randn(1, len(layout.audio_indices), int(cfg.audio_in_channels), generator=g)
    text = torch.randn(1, args.text_tokens, int(cfg.text_dim), generator=g)
    timestep = torch.tensor([0.5, 0.5])
    ts_idx = torch.ones(layout.sequence_length, dtype=torch.int64)
    ts_idx[layout.audio_indices] = 0
    if Path(mod_cache).exists():
        table = torch.load(mod_cache)
    else:
        table = build_adaln_modulation_table(
            transformer_path, timestep[None], num_layers=int(cfg.num_layers), freq_dim=int(cfg.freq_dim),
            time_embed_hidden_dim=int(cfg.time_embed_hidden_dim), time_embed_dim=int(cfg.time_embed_dim),
            hidden_size=int(cfg.hidden_size), dtype=torch.bfloat16)[0]
        torch.save(table, mod_cache)
    return dict(
        hidden_states=video.bfloat16(), audio_hidden_states=audio.bfloat16(),
        encoder_hidden_states=text.bfloat16(), timestep=timestep, timestep_indices=ts_idx,
        token_tags=layout.token_tags.to(torch.int64), position_ids=layout.position_ids.float(),
        video_indices=layout.video_indices, audio_indices=layout.audio_indices,
        text_indices=layout.text_indices, block_modulation=table,
    ), layout.sequence_length


def _worker(rank, world, args, reply_q):
    try:
        from difflet.backends.tpu.ops_impl import device_runtime

        device_runtime.initialize_replica(rank, world)
        import torch
        from accelerate import init_empty_weights

        from difflet.backends.tpu.core.checkpoint import load_windowed_checkpoint_into
        from difflet.backends.tpu.core.weights import materialize_meta_
        from difflet.backends.tpu.ops_impl import parallel_mesh
        from difflet.backends.tpu.ops_impl.platform import configure_matmul_precision
        from difflet.models.minimax_h3.modeling_minimax_h3 import MiniMaxH3Transformer3DModel
        from difflet.pipeline.parallel_mesh import MeshSpec

        configure_matmul_precision()
        parallel_mesh.init_parallel_mesh(MeshSpec(tp=world))
        dev = device_runtime.device()
        path = _transformer_path()
        cfg_dict = MiniMaxH3Transformer3DModel.load_config(path)
        with init_empty_weights(include_buffers=False):
            model = MiniMaxH3Transformer3DModel.from_config(cfg_dict, precomputed_adaln=True)
            _shard_for_tp(model, world)
        materialize_meta_(model, dtype=torch.bfloat16)
        mark = time.monotonic()
        report = load_windowed_checkpoint_into(
            model, path, tp_size=world, tp_rank=rank, dtype=torch.bfloat16, strict=True,
            rename=_rename(int(model.config.ffn_dim)))
        load_s = time.monotonic() - mark
        unexpected = [k for k in report.get("unexpected", []) if ".adaln_proj." not in k]
        model = model.eval().requires_grad_(False).to(dev)
        inputs, seq = _inputs(args, model.config, path, args.mod_cache)
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        def hbm():
            device_runtime.wait()
            return round(device_runtime.memory_info()["bytes_used"] / 2**30, 2)

        def run(module, label):
            mark = time.monotonic()
            with torch.no_grad():
                out = module(**inputs, return_dict=False)
            float(out[0][0, 0, 0].cpu())
            first = time.monotonic() - mark
            times = []
            for _ in range(args.iters):
                mark = time.monotonic()
                with torch.no_grad():
                    out = module(**inputs, return_dict=False)
                float(out[0][0, 0, 0].cpu())
                times.append(time.monotonic() - mark)
            return {"label": label, "first_s": round(first, 2), "warm_ms": [round(t * 1e3, 1) for t in times],
                    "hbm_gib": hbm()}, (out[0].float().cpu(), out[1].float().cpu())

        results = {"rank": rank, "seq": seq, "load_s": round(load_s, 1), "unexpected": unexpected[:5],
                   "hbm_loaded_gib": hbm()}
        compiled = device_runtime.maybe_compile(model, default=args.compile)
        results["bf16"], ref = run(compiled, "bf16")
        if args.group < 0:
            # Noise floor: the same bf16 weights, eager instead of compiled -- two
            # valid bf16 executions of one model.
            results["int8"], got = run(model, "bf16 eager (noise floor)")
        else:
            n = quantize_block_linears_(model, group=args.group)
            torch._dynamo.reset()
            compiled = device_runtime.maybe_compile(model, default=args.compile)
            results["int8_linears"] = n
            results["hbm_quantized_gib"] = hbm()
            label = f"int8 weight-only, {'group ' + str(args.group) if args.group else 'per row'}"
            results["int8"], got = run(compiled, label)
        errs = {}
        for name, a, b in zip(("video", "audio"), got, ref):
            errs[name] = {
                "rel_l2": float((a - b).norm() / b.norm()),
                "cos": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
                "max_abs": float((a - b).abs().max()),
                "finite": bool(torch.isfinite(a).all()),
            }
        results["int8_vs_bf16"] = errs
        reply_q.put(results)
    except BaseException as exc:  # noqa: BLE001
        import traceback

        reply_q.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-3000:]})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--compile", type=int, default=1)
    parser.add_argument("--group", type=int, default=0, help="int8 scale group size (0: per row; -1: no int8, compare bf16 eager vs compiled)")
    parser.add_argument("--mod-cache", default="/mnt/models/profile/h3_modulation_t05.pt")
    args = parser.parse_args()

    import importlib.metadata as md

    vers = {p: md.version(p) for p in ("torch", "torch_tpu", "libtpu", "jax", "diffusers") if _has(md, p)}
    print("versions", vers, "args", vars(args), flush=True)
    # Build the modulation table once, before the workers (13B adaln weights, host).
    if not Path(args.mod_cache).exists():
        import torch  # noqa: F401

        from difflet.models.minimax_h3.modeling_minimax_h3 import MiniMaxH3Transformer3DModel

        cfg = MiniMaxH3Transformer3DModel.load_config(_transformer_path())
        mark = time.monotonic()
        _inputs(args, type("C", (), cfg)(), _transformer_path(), args.mod_cache)
        print(f"modulation table built in {time.monotonic() - mark:.1f}s", flush=True)

    from difflet.backends.tpu.ops_impl import device_runtime

    device_runtime.prepare_launch(args.world)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, args.world, args, reply_q)) for r in range(args.world)]
    for p in procs:
        p.start()
    results = [reply_q.get() for _ in procs]
    for p in procs:
        p.join()
    ok = True
    for res in sorted(results, key=lambda r: r["rank"]):
        if "error" in res:
            ok = False
            print(f"rank {res['rank']} FAILED: {res['error']}\n{res['tb']}", flush=True)
        elif res["rank"] == 0:
            for key, value in res.items():
                print(f"{key}: {value}", flush=True)
    return 0 if ok else 1


def _has(md, name):
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    sys.exit(main())
