"""Manual check: the neuron LTX-2 denoise against a Trainium AoT run, from the same inputs.

The two backends' venvs ship different torch builds, and their bf16 ``torch.randn`` differ for
the same seed (torch 2.9.1 vs 2.14.0: the first element of the seed-42 video noise is -0.809 vs
1.930), so a same-seed generate starts from different noise on each and the videos differ for
that reason alone. This check removes the RNG and the host text encoder from the comparison: it
runs ``TorchNeuronLtx2Application.generate`` on torchrun ranks with the AoT run's own initial
noise (``--noise``: the AoT torch build's seed-42 draws, video then audio, as the orchestrator
draws them) and its own conditioning (``--conditioning``: the A/B probe's ``conditioning.pt``
from the AoT generate), the AoT run's steps and guidance, and compares the final packed latents
with the AoT run's (``--aot-latents``: the probe's ``latents.pt``): cosine and rel_l1 per stream.

    HF_HUB_OFFLINE=1 DIFFLET_COMPILE_CACHE=<cache> PYTHONPATH=<worktree> <venv>/bin/torchrun \\
        --standalone --nproc-per-node 4 <worktree>/tests/manual/check_neuron_ltx2_e2e_parity.py \\
        --model-dir <snapshot> --noise rng_aot.pt --conditioning <aot run>/saved/conditioning.pt \\
        --aot-latents <aot run>/saved/latents.pt --out-json result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["DIFFLET_BACKEND"] = "neuron"

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--noise", type=Path, required=True)
    p.add_argument("--conditioning", type=Path, required=True)
    p.add_argument("--aot-latents", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=704)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    from difflet.backends.registry import get_backend
    from difflet.models.ltx_2.neuron_application import TorchNeuronLtx2Application
    from difflet.models.ltx_2.pipeline import pack_ltx_2_audio_latents, pack_ltx_2_video_latents
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    noise = torch.load(args.noise, weights_only=False)
    cond = torch.load(args.conditioning, weights_only=False)
    aot = torch.load(args.aot_latents, weights_only=False)
    app = TorchNeuronLtx2Application(
        model_path=str(args.model_dir), parallel=parallel, dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width, "num_frames": args.num_frames},
        enable_text_encoder=False, device="neuron",
    )
    t0 = time.perf_counter()
    app.load()
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = app.generate(
        encoder_hidden_states=cond[0], audio_encoder_hidden_states=cond[1],
        encoder_attention_mask=cond[2], audio_encoder_attention_mask=cond[3],
        latents=pack_ltx_2_video_latents(noise["video"]),
        audio_latents=pack_ltx_2_audio_latents(noise["audio"]),
        num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
        output_type="latent",
    )
    gen_s = time.perf_counter() - t0
    result = {"rank": rank, "torch_noise": noise["torch"], "load_s": load_s, "generate_s": gen_s}
    for key, mine in (("video", out.latents), ("audio", out.audio_latents)):
        a, b = mine.double().flatten(), aot[key].double().flatten()
        result[key] = {
            "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
            "rel_l1": float((a - b).abs().mean() / b.abs().mean()),
            "std_neuron": float(a.std()), "std_aot": float(b.std()),
        }
    if rank == 0:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2))
        if args.save is not None:
            torch.save({"video": out.latents, "audio": out.audio_latents}, args.save)
        print(f"[parity] {json.dumps(result)}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
