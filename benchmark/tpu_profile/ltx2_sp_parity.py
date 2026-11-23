"""LTX-2 SP vs TP, piece by piece on one block, on the chips (tp=4, eager).

    python -m benchmark.tpu_profile.ltx2_sp_parity

Builds the TP-sharded DiT (480x704x49: 2310 video rows, padded to 2312 under
SP), captures block 0's real inputs from one forward on random latents, then
runs each video-facing piece both ways on the same input and prints the
relative L2 of SP against TP (rank 0):

* attn1  video self-attention        (AG q/k/v, pad keys masked, RS out)
* attn2  video-text cross-attention   (AG q, RS out)
* a2v    audio-to-video               (AG q, RS out)
* v2a    video-to-audio               (AG k/v, pad keys masked, TP out)
* ff     video feed-forward           (AG, RS)
* block  the whole block

TP is the reference path (the shared processors); bf16 differences are ~1e-2,
a broken piece is ~1.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import types
from pathlib import Path

MODEL = os.environ.get("DIFFLET_LTX2_PATH", "/mnt/models/hf/hub/models--Lightricks--LTX-2/snapshots")


def _model_path() -> str:
    path = Path(MODEL)
    if (path / "transformer").is_dir():
        return str(path)
    return str(next(p for p in sorted(path.iterdir()) if (p / "transformer").is_dir()))


def _worker(rank, world, reply_q):
    try:
        os.environ["DIFFLET_TPU_SP"] = "0"  # build plain TP; SP pieces are called by hand
        from difflet.backends.tpu.ops_impl import device_runtime

        device_runtime.initialize_replica(rank, world)
        import torch

        from difflet.backends.tpu.ltx_2 import sp
        from difflet.backends.tpu.ltx_2.config import TpuLTX2Config
        from difflet.backends.tpu.ltx_2.transformer import TpuLTX2TransformerApplication
        from difflet.backends.tpu.ops_impl import collective_matmul as cm
        from difflet.backends.tpu.ops_impl import parallel_mesh
        from difflet.ops import gather_from_sequence_parallel_region as gather

        path = os.path.join(_model_path(), "transformer")
        hh, ww, ff_ = (int(v) for v in os.environ.get("PARITY_SHAPE", "480x704x49").split("x"))
        cfg = TpuLTX2Config.from_pretrained(path, height=hh, width=ww, num_frames=ff_,
                                            tp_degree=world, torch_dtype=torch.bfloat16)
        app = TpuLTX2TransformerApplication(model_path=path, config=cfg)
        app._init_runtime()
        dev = device_runtime.device()
        module = app._prepare_module().to(dev)
        app.module = module
        tp, tp_rank = parallel_mesh.get_tp_size(), parallel_mesh.get_tp_rank()

        captured = {}

        def grab(_mod, args, kwargs):
            if not captured:
                captured.update(kwargs)

        block = module.transformer_blocks[0]
        handle = block.register_forward_pre_hook(grab, with_kwargs=True)
        g = torch.Generator().manual_seed(7)
        inputs = [t.normal_(generator=g) if t.is_floating_point() and i < 4 else t
                  for i, t in enumerate(app.get_example_inputs())]
        with torch.no_grad():
            app.forward(*[t.to(dev) for t in inputs])
        handle.remove()
        kw = captured
        seq = int(cfg.video_seq_len)
        padded = sp.padded_length(seq, tp)
        n = padded // tp
        if os.environ.get("PARITY_PLAIN", "1") == "1":
            # The rings hang under eager torch_tpu (every rank stuck reading the
            # first ring result back, 2026-11-10); their meaning with plain
            # collectives checks sp.py's own logic.
            import torch.nn.functional as F

            from difflet.ops import reduce_scatter_to_sequence_parallel_region as rs

            def plain_ag(x, linears):
                full = gather(x.contiguous(), dim=1)
                return [F.linear(full, lin.weight, lin.bias) for lin in linears]

            def plain_rs(x, linear):
                out = rs(F.linear(x, linear.weight), dim=1)
                return out if linear.bias is None else out + linear.bias.to(out.dtype)

            cm.all_gather_matmul = plain_ag
            cm.matmul_reduce_scatter = plain_rs
        else:
            cm.prepare(sp.LTX2_OPS)
        if rank == 0:
            print("[parity] captured block 0 inputs; rings ready", flush=True)

        def shard(x):
            # .clone(): a jax_op input at a nonzero storage offset hangs the ring
            # (torch_tpu 0.2.0, see collective_matmul's docstring).
            return sp._pad_seq(x, 1, padded, 0.0)[:, tp_rank * n:(tp_rank + 1) * n].clone()

        def step(name):
            if rank == 0:
                print(f"[parity] {name}", flush=True)

        def unshard(x):
            return gather(x, dim=1)[:, :seq]

        def pad_rope(r):
            cos, sin = r
            d = 2 if cos.ndim == 4 else 1
            return sp._pad_seq(cos, d, padded, 1.0), sp._pad_seq(sin, d, padded, 0.0)

        def rel(a, b):
            a, b = a.float().cpu(), b.float().cpu()
            return round(float((a - b).norm() / b.norm().clamp_min(1e-12)), 5)

        h = kw["hidden_states"]
        audio = kw["audio_hidden_states"]
        x = block.norm1(h)  # any per-token video input will do
        xa = block.audio_norm1(audio)
        vrope, crope = pad_rope(kw["video_rotary_emb"]), pad_rope(kw["ca_video_rotary_emb"])
        res = {}
        if os.environ.get("PARITY_PLAIN", "1") != "1":
            # Ring mode: each site's AG ring, compiled, on the model's real input
            # and weights, against gather + linear. Two inputs: a fresh shard
            # (clone), and the shard taken inside the compiled function the way
            # the first block does (pad, slice, norm).
            import torch.nn.functional as F

            sites = {
                "attn1": (block.attn1.to_q, block.attn1.to_k, block.attn1.to_v),
                "attn2": (block.attn2.to_q,),
                "a2v": (block.audio_to_video_attn.to_q,),
                "v2a": (block.video_to_audio_attn.to_k, block.video_to_audio_attn.to_v),
                "ff": (block.ff.net[0].proj,),
            }
            with torch.no_grad():
                full = gather(shard(x).contiguous(), dim=1)
                for name, lins in (sites.items() if os.environ.get("PARITY_SITES") == "1" else ()):
                    step(name)
                    want = [F.linear(full, lin.weight, lin.bias) for lin in lins]
                    ring = torch.compile(lambda t, lins=lins: cm.all_gather_matmul(t, lins), backend="tpu")
                    got = ring(shard(x))
                    res[name] = [rel(g_, w_) for g_, w_ in zip(got, want)]

                    def inside(hh, lins=lins):
                        t = sp._pad_seq(hh, 1, padded, 0.0)[:, tp_rank * n:(tp_rank + 1) * n]
                        return cm.all_gather_matmul(block.norm1(t), lins)

                    ref_in = gather(block.norm1(shard(h)).contiguous(), dim=1)
                    want2 = [F.linear(ref_in, lin.weight, lin.bias) for lin in lins]
                    got2 = torch.compile(inside, backend="tpu")(h)
                    res[name + "_sliced"] = [rel(g_, w_) for g_, w_ in zip(got2, want2)]
                ns = dict(seq=seq, padded=padded, tp=tp, rank=tp_rank, chain=False)
                kw2 = dict(kw, video_rotary_emb=vrope, ca_video_rotary_emb=crope)
                ref_h, ref_a = block(**kw)
                ref_shard = shard(ref_h)

                def say(msg):
                    if rank == 0:
                        print(f"[parity] {msg}", flush=True)

                b1 = module.transformer_blocks[1]
                r_h, r_a = b1(**dict(kw, hidden_states=ref_h, audio_hidden_states=ref_a))
                block._difflet_sp = types.SimpleNamespace(first=True, last=False, **ns)
                b1._difflet_sp = types.SimpleNamespace(first=False, last=True, **ns)

                def two(**k):
                    h2, a2 = sp._block_forward(block, **k)
                    return sp._block_forward(b1, **dict(k, hidden_states=h2, audio_hidden_states=a2))

                g_h, g_a = torch.compile(two, backend="tpu")(**kw2)
                res["two_video"] = rel(g_h, r_h)
                res["two_audio"] = rel(g_a, r_a)
                if rank == 0:
                    print(f"[parity] H shape {hh}x{ww}x{ff_} seq {seq} padded {padded}: two blocks "
                          f"video {res['two_video']} audio {res['two_audio']}", flush=True)
            reply_q.put({"rank": rank, "tp_rank": tp_rank, "mode": "ring", **res})
            return
        res = {}
        with torch.no_grad():
            step("attn1")
            a = block.attn1
            ref = a(x, encoder_hidden_states=None, query_rotary_emb=kw["video_rotary_emb"])
            q, k, v = cm.all_gather_matmul(shard(x), (a.to_q, a.to_k, a.to_v))
            got = unshard(sp._out_rs(a, sp._attend(a, q, k, v, q_rope=vrope, valid_keys=seq)))
            res["attn1"] = rel(got, ref)

            step("attn2")
            a = block.attn2
            enc, mask = kw["encoder_hidden_states"], kw["encoder_attention_mask"]
            ref = a(x, encoder_hidden_states=enc, attention_mask=mask)
            (q,) = cm.all_gather_matmul(shard(x), (a.to_q,))
            got = unshard(sp._out_rs(a, sp._attend(a, q, a.to_k(enc), a.to_v(enc), text_mask=mask)))
            res["attn2"] = rel(got, ref)

            step("a2v")
            a = block.audio_to_video_attn
            ref = a(x, encoder_hidden_states=xa, query_rotary_emb=kw["ca_video_rotary_emb"],
                    key_rotary_emb=kw["ca_audio_rotary_emb"])
            (q,) = cm.all_gather_matmul(shard(x), (a.to_q,))
            got = unshard(sp._out_rs(a, sp._attend(a, q, a.to_k(xa), a.to_v(xa), q_rope=crope,
                                                   k_rope=kw["ca_audio_rotary_emb"])))
            res["a2v"] = rel(got, ref)

            step("v2a")
            a = block.video_to_audio_attn
            ref = a(xa, encoder_hidden_states=x, query_rotary_emb=kw["ca_audio_rotary_emb"],
                    key_rotary_emb=kw["ca_video_rotary_emb"])
            k, v = cm.all_gather_matmul(shard(x), (a.to_k, a.to_v))
            got = a.to_out[1](a.to_out[0](sp._attend(a, a.to_q(xa), k, v,
                                                     q_rope=kw["ca_audio_rotary_emb"], k_rope=crope,
                                                     valid_keys=seq)))
            res["v2a"] = rel(got, ref)

            step("ff")
            ref = block.ff(x)
            got = unshard(sp._ff(block.ff, shard(x)))
            res["ff"] = rel(got, ref)

            step("block")
            ref_h, ref_a = block(**kw)
            block._difflet_sp = types.SimpleNamespace(seq=seq, padded=padded, tp=tp, rank=tp_rank,
                                                      first=True, last=True, chain=False)
            # SP takes the video stream padded outside the graph (pad_video_inputs).
            kw2 = dict(kw, hidden_states=sp._pad_seq(kw["hidden_states"], 1, padded, 0.0),
                       video_rotary_emb=vrope, ca_video_rotary_emb=crope)
            got_h, got_a = sp._block_forward(block, **kw2)
            got_h = got_h[:, :seq]
            res["block_video"] = rel(got_h, ref_h)
            res["block_audio"] = rel(got_a, ref_a)
        reply_q.put({"rank": rank, "tp_rank": tp_rank, "seq": seq, "padded": padded,
                     "rope_shape": tuple(kw["video_rotary_emb"][0].shape), **res})
    except BaseException as exc:  # noqa: BLE001
        import traceback

        reply_q.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}",
                     "tb": traceback.format_exc()[-2500:]})
        raise


def main() -> int:
    from difflet.backends.tpu.ops_impl import device_runtime

    world = 4
    device_runtime.prepare_launch(world)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, reply_q)) for r in range(world)]
    for p in procs:
        p.start()
    results = sorted((reply_q.get(timeout=2400) for _ in procs), key=lambda r: r["rank"])
    for p in procs:
        p.join(timeout=60)
    for r in results:
        print(r, flush=True)
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
