"""Device check: Wan's VAE decoder on the neuron backend against diffusers' host decode.

Run under torchrun with 4 ranks, from the repository root:

    HF_HUB_OFFLINE=1 PYTHONPATH=. torchrun --standalone --nproc-per-node 4 \\
        tests/manual/check_neuron_wan_vae.py --model-dir <Wan snapshot> --out-dir <dir> \\
        [--num-frames 9]

The snapshot needs vae/. Every rank decodes one seeded latent with
``NeuronWanVaeDecoderApplication`` (fp32, eager) through ``_DeviceVaeDecoder``, the callable the
application hands the orchestrator. Rank 0 then decodes the same latent with diffusers'
AutoencoderKLWan in fp32 on the host (the CLI's ``--host-vae`` path) with a pinned thread count
and judges the device frames fp32 against fp32: ``rel_max`` within ``FP32_MAX_REL`` of the
reference's peak, and the cosine floor. ``--num-frames 81`` (21 latent frames) is the clip
length the AoT path has to decode on the host.

Checks: parity, identical frames on all ranks, no CPU fallbacks. Reported: decode seconds
(first decode after the warm-up, then a repeat), peak device memory, load phases, and the host
reference's decode seconds. Each rank writes ``result-rank<r>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from difflet.backends.neuron.compile import configure_compile_cache

configure_compile_cache()  # before torch: importing torch autoloads torch_neuronx

import torch  # noqa: E402

from difflet.backends.neuron.runtime import track_fallbacks  # noqa: E402
from difflet.backends.neuron.wan.vae import NeuronWanVaeDecoderApplication  # noqa: E402
from difflet.backends.registry import get_backend  # noqa: E402
from difflet.models.wan.neuron_application import _DeviceVaeDecoder  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.numerical import neuron_parity as parity  # noqa: E402
from tests.numerical.buildinfo import build_info  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference-threads", type=int, default=12)
    args = p.parse_args()
    if not (args.model_dir / "vae").is_dir():
        p.error(f"{args.model_dir} has no vae/")
    if (args.num_frames - 1) % 4 or args.height % 8 or args.width % 8:
        p.error("num_frames must be 4k+1 and height/width multiples of 8")
    return args


def timed_decode(decoder, z) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    frames = decoder(z)
    return frames, time.perf_counter() - start


def main() -> int:
    args = parse_args()
    import torch_neuronx.metrics as metrics

    metrics.set_enabled(True)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vae_dir = args.model_dir / "vae"
    with open(vae_dir / "config.json", encoding="utf-8") as f:
        z_dim = int(json.load(f)["z_dim"])
    latent_shape = (1, z_dim, (args.num_frames - 1) // 4 + 1, args.height // 8, args.width // 8)
    app = NeuronWanVaeDecoderApplication(model_path=str(vae_dir), latent_shape=latent_shape,
                                         parallel=parallel, device="neuron")
    with track_fallbacks() as load_fallbacks:
        app.load()
    decoder = _DeviceVaeDecoder(app, device="neuron")
    z = torch.randn(latent_shape, generator=torch.Generator().manual_seed(args.seed))

    with track_fallbacks() as fallbacks:
        frames, first_s = timed_decode(decoder, z)
        _, repeat_s = timed_decode(decoder, z)
    peak = (torch.neuron.max_memory_allocated() / 2**30
            if hasattr(torch.neuron, "max_memory_allocated") else None)
    result = {
        "rank": rank, "world_size": world, "latent_shape": list(latent_shape),
        "frames_shape": list(frames.shape), "decode_seconds": {"first": first_s, "repeat": repeat_s},
        "fallbacks": [*load_fallbacks, *fallbacks], "peak_device_mem_gb": peak,
        "phase_seconds": app.phase_seconds, "load_report": {k: len(v) for k, v in app.load_report.items()},
        "frames_sha256": hashlib.sha256(frames.contiguous().numpy().tobytes()).hexdigest(),
        "build": build_info(),
    }
    ok = not result["fallbacks"] and result["load_report"].get("missing", 0) == 0
    if rank == 0:
        from diffusers import AutoencoderKLWan

        torch.set_num_threads(args.reference_threads)
        vae = AutoencoderKLWan.from_pretrained(str(vae_dir), torch_dtype=torch.float32).eval()
        start = time.perf_counter()
        with torch.no_grad():
            reference = vae.decode(z, return_dict=False)[0]
        result["host_decode_seconds"] = time.perf_counter() - start
        m = parity.metrics(frames, reference)
        passed, reasons = parity.check(m)
        if not m["rel_max"] <= parity.FP32_MAX_REL:
            passed = False
            reasons.append(f"rel_max {m['rel_max']:.3e} above FP32_MAX_REL {parity.FP32_MAX_REL}")
        result["parity"] = {"ok": passed, "reasons": reasons, "metrics": m}
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL'} parity frames {tuple(frames.shape)} "
              f"cos={m['cosine']:.8f} rel_l1={m['rel_l1']:.3e} rel_max={m['rel_max']:.3e} "
              f"{reasons} | host decode {result['host_decode_seconds']:.1f}s", flush=True)
    (args.out_dir / f"result-rank{rank}.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[rank {rank}] fallbacks={result['fallbacks']} load={result['load_report']} "
          f"decode s: first={first_s:.2f} repeat={repeat_s:.2f} peak_device_mem_gb={peak} "
          f"phases={ {k: round(v, 1) for k, v in app.phase_seconds.items()} }", flush=True)

    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            shas = [json.loads((args.out_dir / f"result-rank{r}.json").read_text())["frames_sha256"]
                    for r in range(world)]
            identical = len(set(shas)) == 1
            ok = ok and identical
            print(f"[rank 0] frames identical on all {world} ranks: {identical}", flush=True)
        dist.destroy_process_group()
    if rank == 0:
        print(f"{'PASS' if ok else 'FAIL'} | build: {result['build']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
