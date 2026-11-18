"""FLUX's T5-XXL on the chips (layer pipeline) against the host fp32 encoder.

    python -m benchmark.tpu_profile.t5_device_probe [--dtypes bf16,fp32]

One process per chip, as in serving. Per dtype, every rank builds
``TpuPipelinedT5Encoder`` (its segment of the 24 blocks), encodes the prompts
twice (the first call compiles) and rank 0 compares each result against the
stock host encoder in fp32 — what ``TpuBroadcastT5Encoder`` computes today.
Reports the warm encode latency, the host latency, and per prompt the cosine
and the max / relative error of the (1, 512, 4096) prompt embeds.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

MODEL = os.environ.get(
    "DIFFLET_FLUX_PATH",
    "/mnt/models/hf/hub/models--black-forest-labs--FLUX.1-dev/snapshots",
)
PROMPTS = [
    "A cat holding a sign that says hello world",
    "A photorealistic portrait of an old fisherman at dawn, golden light, 85mm lens, shallow depth of field",
    "",
]


def _model_path() -> str:
    path = Path(MODEL)
    if (path / "text_encoder_2").is_dir():
        return str(path)
    return str(next(p for p in sorted(path.iterdir()) if (p / "text_encoder_2").is_dir()))


def _worker(rank, world, dtypes, reply_q):
    try:
        from difflet.backends.tpu.ops_impl import device_runtime

        device_runtime.initialize_replica(rank, world)
        import torch

        from difflet.models.flux.tpu_application import (
            FLUX_TEXT_SEQ_LEN,
            TpuBroadcastT5Encoder,
            TpuPipelinedT5Encoder,
            _normalize_dtype,
        )

        path = _model_path()
        results: dict = {"rank": rank}
        if rank == 0:
            ref = TpuBroadcastT5Encoder(path, seq_len=FLUX_TEXT_SEQ_LEN, dtype=torch.float32)
            ref_out, host_s = [], []
            for prompt in PROMPTS:
                mark = time.monotonic()
                ref_out.append(ref.encode_local(prompt).float())
                host_s.append(time.monotonic() - mark)
            results["host_fp32_s"] = [round(s, 3) for s in host_s]
            del ref
        for name in dtypes:
            enc = TpuPipelinedT5Encoder(path, seq_len=FLUX_TEXT_SEQ_LEN, dtype=torch.float32,
                                        encoder_dtype=_normalize_dtype(name))
            mem_gb = device_runtime.memory_info()["bytes_used"] / 2**30
            first = []
            for prompt in PROMPTS:
                mark = time.monotonic()
                enc(prompt)
                first.append(time.monotonic() - mark)
            warm, outs = [], []
            for prompt in PROMPTS:
                mark = time.monotonic()
                outs.append(enc(prompt).float())
                warm.append(time.monotonic() - mark)
            row = {"first_s": [round(s, 3) for s in first], "warm_s": [round(s, 3) for s in warm],
                   "hbm_used_gb": round(mem_gb, 2)}
            if rank == 0:
                errs = []
                for got, want in zip(outs, ref_out):
                    diff = (got - want).abs()
                    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
                    errs.append({"cos": round(float(cos), 6), "max_abs": round(float(diff.max()), 4),
                                 "rel": round(float(diff.norm() / want.norm()), 5)})
                row["vs_host_fp32"] = errs
            results[name] = row
            del enc
            import gc

            gc.collect()
        reply_q.put(results)
    except Exception as exc:  # noqa: BLE001 - report, the parent prints it
        import traceback

        traceback.print_exc()
        reply_q.put({"rank": rank, "error": repr(exc)})


def main() -> None:
    from difflet.backends.tpu.ops_impl import device_runtime

    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", default="bf16,fp32")
    args = parser.parse_args()
    dtypes = [d for d in args.dtypes.split(",") if d]
    world = len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    device_runtime.prepare_launch(world)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, dtypes, reply_q)) for r in range(world)]
    for p in procs:
        p.start()
    for _ in range(world):
        msg = reply_q.get(timeout=1500)
        print(msg, flush=True, file=sys.stdout if msg.get("rank") == 0 else sys.stderr)
    for p in procs:
        p.join(timeout=60)


if __name__ == "__main__":
    main()
