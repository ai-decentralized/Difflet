"""Stage 2: difflet on TPU against the diffusers reference from stage 1.

Runs the same fixed inputs through difflet's sharded TPU model and reports
cosine / relative error against the upstream fp32 result. Runs it twice -- with
the fused Pallas attention and with SDPA -- so the question "did switching
kernels cost accuracy?" is answered by measurement rather than by the op-level
microbenchmark alone.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path


def _metrics(actual, reference):
    import torch

    a = actual.float().flatten()
    r = reference.float().flatten()
    cos = float(torch.nn.functional.cosine_similarity(a, r, dim=0))
    rel = float((a - r).abs().mean() / r.abs().mean())
    rel_l2 = float((a - r).norm() / r.norm())
    return {"cosine": cos, "rel_l1": rel, "rel_l2": rel_l2,
            "max_abs": float((a - r).abs().max()),
            "ref_absmean": float(r.abs().mean())}


def worker(rank, world, model, ref_path, q):
    try:
        _worker(rank, world, model, ref_path, q)
    except Exception as exc:  # noqa: BLE001
        import traceback
        q.put({"done": rank, "error": repr(exc), "tb": traceback.format_exc()})
        raise


def _worker(rank, world, model, ref_path, q):
    from torch_xla._internal import pjrt

    pjrt.initialize_multiprocess(rank, world)

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    torch.set_num_threads(max(1, (os.cpu_count() or world) // world))
    device = torch_xla.device()
    blob = torch.load(ref_path, weights_only=False)
    reference = blob["reference"]
    # The same upstream implementation in bf16. Comparing against fp32 alone
    # conflates "difflet is wrong" with "bf16 costs this much"; this separates
    # them.
    reference_bf16 = blob.get("reference_bf16")

    from difflet.pipeline.parallel_config import DiffletParallelConfig

    if model == "wan":
        from difflet.models.wan.entry import create_wan_application

        frames = int(blob.get("frames", 9))
        app = create_wan_application(
            model_path=WAN, parallel=DiffletParallelConfig(tp_degree=world),
            dtype=torch.bfloat16,
            shape={"height": 480, "width": 832, "num_frames": frames},
            backend="tpu",
        )
        holder = app.transformer
        inputs = (blob["hidden_states"], blob["timestep"],
                  blob["encoder_hidden_states"])
    else:
        from difflet.models.qwen_image.entry import create_qwen_image_application

        app = create_qwen_image_application(
            model_path=QWEN, parallel=DiffletParallelConfig(tp_degree=world),
            dtype=torch.bfloat16,
            shape={"height": 1024, "width": 1024, "num_frames": None},
            backend="tpu", text_seq_len=1024,
        )
        holder = app.transformer
        inputs = (blob["hidden_states"], blob["timestep"],
                  blob["encoder_hidden_states"], None, None)

    module = holder._prepare_module().to(device)
    xm.mark_step()
    xm.wait_device_ops()

    from difflet.backends.tpu.ops_impl import attention as A

    results = {}
    for label, flash in (("fused", True), ("sdpa", False)):
        A._FLASH_KERNEL = None
        os.environ["DIFFLET_TPU_FLASH"] = "1" if flash else "0"
        available = A._flash_kernel() is not None
        with torch.no_grad():
            out = module(*[t.to(torch.bfloat16).to(device) if t is not None else None
                           for t in inputs])
            xm.mark_step()
            out = out.cpu()
        if rank == 0:
            entry = {"kernel_available": available, **_metrics(out, reference)}
            if reference_bf16 is not None:
                entry["vs_bf16"] = _metrics(out, reference_bf16)
            results[label] = entry
    if rank == 0:
        control = (_metrics(reference_bf16, reference)
                   if reference_bf16 is not None else None)
        q.put({"model": model, "results": results, "control": control,
               "shape": list(reference.shape)})
    q.put({"done": rank})


QWEN = os.environ.get(
    "DIFFLET_QWEN_SNAPSHOT",
    "/mnt/models/hf/hub/models--Qwen--Qwen-Image/snapshots/"
    "75e0b4be04f60ec59a75f475837eced720f823b6",
)
WAN = os.environ.get(
    "DIFFLET_WAN_SNAPSHOT",
    "/mnt/models/hf/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/"
    "snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["wan", "qwen_image"])
    p.add_argument("--ref", default=None)
    a = p.parse_args()
    ref = a.ref or f"/mnt/models/oracle_{a.model}.pt"

    world = len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, a.model, ref, q))
             for r in range(world)]
    for proc in procs:
        proc.start()
    seen, payload = 0, None
    while seen < world:
        if not any(x.is_alive() for x in procs) and q.empty():
            print("workers exited early", flush=True)
            break
        try:
            msg = q.get(timeout=1800)
        except Exception:  # noqa: BLE001
            continue
        if "done" in msg:
            seen += 1
            if msg.get("tb"):
                print(msg["tb"], flush=True)
        else:
            payload = msg
    for proc in procs:
        proc.join(timeout=120)
        if proc.is_alive():
            proc.terminate()

    if payload is None:
        return 1
    print(f"\n{payload['model']}  output {payload['shape']}  "
          f"vs diffusers fp32 on CPU")
    print(f"{'attention':10s} {'cos vs fp32':>13s} {'rel_l1':>9s} | "
          f"{'cos vs bf16':>13s} {'rel_l1':>9s}")
    for label, m in payload["results"].items():
        b = m.get("vs_bf16") or {}
        line = (f"{label:10s} {m['cosine']:13.8f} {m['rel_l1']:9.2e} | "
                f"{b.get('cosine', float('nan')):13.8f} {b.get('rel_l1', float('nan')):9.2e}")
        if not m["kernel_available"] and label == "fused":
            line += "  (kernel unavailable!)"
        print(line)
    ctrl = payload.get("control")
    if ctrl:
        print(f"{'[control]':10s} {ctrl['cosine']:13.8f} {ctrl['rel_l1']:9.2e} | "
              f"{'':13s} {'':9s}  diffusers bf16 vs its own fp32")
    Path(f"/mnt/models/oracle_{payload['model']}_result.json").write_text(
        json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
