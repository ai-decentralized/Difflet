"""Could the primary's DiT shard wait out the VAE decode on a peer chip instead of the host?

``device_runtime.vae_decode_room`` moves the primary replica's DiT shard (~7 GB
for Wan at tp=4) to the host and back around every chip VAE decode: 1.5-2.4 s
each way in ``difflet serve`` (4-5 GB/s), on the critical path. The other three
chips sit idle during the decode, so their HBM could hold it instead -- over
ICI rather than the host link. Two things decide whether that works:

1. Room: free HBM on every rank in the serving state (Wan loaded, one request
   run, the primary's VAE on its chip).
2. Speed: rank 0 -> rank 1 (and back) over ICI, against the host round trip of
   the same bytes measured here -- one contiguous buffer, so the gap to the
   per-parameter ``module.to`` in serve (~1000 tensors) shows up too.

First run (2026-10-28 21:29 UTC): ``dist.send``/``dist.recv`` hung on torch_tpu
-- the sender sat in ``send`` while the receiver had already returned and
reached the next barrier (py-spy dumps in profile/peer-stash-*/hang_dump_*).
Second run (21:43 UTC): ``dist.new_group([0, 1])`` hung in
``_new_process_group_helper`` on ranks 0/1 (2/3 passed to the barrier).
Third run (21:47 UTC): world-group ``dist.broadcast`` hung too -- the source
returned, the receivers sat in ``device_runtime.wait()``. The one collective
known to work is the all-reduce every TP layer uses, so a transfer is now
``collectives.reduce_tp`` with zeros on every rank but the source: all ranks
end with the source's bytes, at roughly twice a broadcast's traffic, so the
time is an upper bound. It is out of place (a second buffer per rank), which
caps the sizes below the 6.98 GiB shard; the shard time is extrapolated.
Fourth run (21:50 UTC): the all-reduce hung the same way (source through,
receivers in wait). Suspect: lazy execution dropped the source's collective --
its buffer was a bare ``torch.empty`` and the result was never read. Buffers
are now written on every rank and one element of each result is read back.

    DIFFLET_TPU_DEVICE_VAE=1 \
        python -m benchmark.tpu_profile.peer_stash_probe
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
REVISION = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"
#: Transfer sizes (GiB); the shard size itself is appended at run time.
SIZES_GIB = (0.25, 1.0, 2.0, 3.0)
R = 3


def _say(rank, **kv):
    print({"rank": rank, **kv}, flush=True)


def worker(rank, world, reply_q):
    try:
        reply_q.put(_worker(rank, world))
    except Exception as exc:  # noqa: BLE001 -- the failure is the finding
        reply_q.put({"rank": rank, "error": repr(exc)[:500], "traceback": traceback.format_exc()[-2000:]})


def _worker(rank, world):
    from difflet.backends.tpu.ops_impl import device_runtime

    device_runtime.initialize_replica(rank, world)
    os.environ["DIFFLET_REPLICA_RANK"] = str(rank)

    import torch
    import torch.distributed as dist
    from huggingface_hub import snapshot_download

    from difflet.models.wan.entry import create_wan_application
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from difflet.serving.models.wan import _application_kwargs, _load_host_vae

    torch.set_num_threads(max(1, (os.cpu_count() or world) // world))
    dev = device_runtime.device()
    gib = 2**30
    out = {"rank": rank}

    def free():
        device_runtime.wait()
        return device_runtime.memory_info(dev)["bytes_free"] / gib

    out["free_empty"] = free()
    model_dir = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    app = create_wan_application(
        model_path=model_dir, parallel=DiffletParallelConfig(tp_degree=world), dtype=torch.bfloat16,
        shape={"height": 480, "width": 832, "num_frames": 9}, backend="tpu", **_application_kwargs(),
    )
    app.load_eager()
    out["free_dit"] = free()
    shard = sum(t.numel() * t.element_size()
                for m in app.on_chip_modules() for t in list(m.parameters()) + list(m.buffers()))
    out["dit_shard_gib"] = shard / gib
    vae = None
    if rank == 0:
        vae = device_runtime.maybe_device_vae(_load_host_vae(model_dir))
    out["free_dit_vae"] = free()

    # One short request so the compiled programs and their buffers exist.
    orch = app.pipeline
    text = orch.encode_prompt(prompt="a red fox running through a snowy forest")
    orch(prompt_embeds=text, num_inference_steps=2, guidance_scale=1.0,
         generator=torch.Generator().manual_seed(0), output_type="latent")
    out["free_after_request"] = free()
    out["peak_used_gib"] = device_runtime.memory_info(dev).get("peak_bytes_used", 0) / gib
    _say(rank, stage="memory", **{k: v for k, v in out.items() if k != "rank"})

    dist.barrier()
    transfers = {}
    from difflet.backends.tpu.ops_impl import collectives

    for size in list(SIZES_GIB) + [round(shard / gib, 3)]:
        n = int(size * gib) // 2
        entry = {}
        try:
            # "0->1" / "1->0" name the source; on the world group all ranks receive.
            buf = torch.empty(n, dtype=torch.bfloat16, device=dev)
            device_runtime.wait()
            for direction, (src, dst) in (("0->1", (0, 1)), ("1->0", (1, 0))):
                if size > max(SIZES_GIB):
                    break  # host round trip only: the all-reduce needs 2x the bytes
                times = []
                for _ in range(R):
                    dist.barrier()
                    mark = time.monotonic()
                    buf.fill_(1.0 if rank == src else 0.0)
                    got = collectives.reduce_tp(buf)
                    check = float(got.view(-1)[-1].cpu())  # forces the collective
                    del got
                    dist.barrier()
                    times.append(time.monotonic() - mark)
                entry[direction] = {"s": min(times), "GBps": size * gib / 1e9 / min(times),
                                    "last_elem": check, "times": times}
            if rank == 0:
                host = []
                for _ in range(R):
                    mark = time.monotonic()
                    cpu = buf.to("cpu")
                    t_out = time.monotonic() - mark
                    mark = time.monotonic()
                    buf = cpu.to(dev)
                    device_runtime.wait()
                    host.append((t_out, time.monotonic() - mark))
                    del cpu
                best = min(host, key=sum)
                entry["host"] = {"to_host_s": best[0], "to_chip_s": best[1],
                                 "GBps": [size * gib / 1e9 / best[0], size * gib / 1e9 / best[1]]}
            del buf
            device_runtime.wait()
        except Exception as exc:  # noqa: BLE001
            entry["error"] = repr(exc)[:400]
            entry["traceback"] = traceback.format_exc()[-1200:]
        transfers[f"{size}GiB"] = entry
        if rank == 0:
            _say(rank, stage="transfer", size_gib=size, **entry)
    out["transfers"] = transfers
    out["free_end"] = free()
    del vae
    return out


def main() -> None:
    from difflet.backends.tpu.ops_impl import device_runtime

    world = len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    device_runtime.prepare_launch(world)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, reply_q)) for r in range(world)]
    for p in procs:
        p.start()
    for _ in range(world):
        print(reply_q.get(timeout=3000), flush=True)
    for p in procs:
        p.join(timeout=120)


if __name__ == "__main__":
    main()
