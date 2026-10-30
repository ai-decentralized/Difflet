"""Device check: Wan 2.2 A14B with both experts resident on the neuron backend (TP4).

What the per-expert preflight (check_neuron_wan_preflight.py --expert ...) cannot show: the two
experts together. Run under torchrun with 4 ranks, from the repository root:

    HF_HUB_OFFLINE=1 PYTHONPATH=. torchrun --standalone --nproc-per-node 4 \\
        tests/manual/check_neuron_wan22_dual.py --model-dir <Wan2.2-T2V-A14B snapshot> \\
        --out-dir <dir> [--exec-mode compile] [--steps 10]

The snapshot needs transformer/, transformer_2/, scheduler/ and model_index.json; the prompt
embeddings are seeded random (the text encoder is not loaded: this checks the experts, not the
text path), and the orchestrator's zero negative embeddings serve the CFG pass.

Checks, per rank:

* second_expert_compiles_nothing: ``CompilationCache.TotalCompilations`` does not grow while
  transformer_2 loads and warms up; its blocks have transformer's graph, so every compiled block
  is a cache hit (compile mode; in eager mode the per-op kernels are shared the same way);
* both_experts_ran: the boundary (``boundary_ratio`` x 1000) sent steps to each expert;
* fallbacks, unwarmed_shapes: as in the preflight;
* latents_finite.

Reported, not gated: ``peak_device_mem_gb`` with both experts resident (24 GiB per logical core
at LNC=2), load phases per expert, and DiT forward times per expert. Rank 0 also checks that the
final latents are bit-identical on every rank. Each rank writes ``result-rank<r>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

from difflet.backends.neuron.compile import configure_compile_cache

# Before torch: importing torch autoloads torch_neuronx, which reads the HLO cache location once.
if "--cache-dir" in sys.argv[:-1]:
    configure_compile_cache(sys.argv[sys.argv.index("--cache-dir") + 1])
else:
    configure_compile_cache()

import torch  # noqa: E402

from difflet.backends.neuron.runtime import track_fallbacks  # noqa: E402
from difflet.backends.registry import get_backend  # noqa: E402
from difflet.models.wan.neuron_application import TorchNeuronWanApplication  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.numerical import neuron_parity as parity  # noqa: E402
from tests.numerical.buildinfo import build_info  # noqa: E402

TOTAL_COMPILATIONS = "TotalCompilations"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Wan 2.2 A14B diffusers snapshot (transformer/ and transformer_2/)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--exec-mode", choices=("eager", "compile"), default="compile")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--guidance-scale-2", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="NEFF cache directory (else the Difflet compile cache's)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    args = p.parse_args()
    for sub in ("transformer", "transformer_2", "scheduler"):
        if not (args.model_dir / sub).is_dir():
            p.error(f"{args.model_dir} has no {sub}/: a Wan 2.2 A14B snapshot is required")
    return args


def total_compilations() -> int:
    import torch_neuronx.metrics as metrics

    return metrics.get_counter_value("CompilationCache." + TOTAL_COMPILATIONS) or 0


def main() -> int:
    args = parse_args()
    # Before any device work: the runtime's compilation cache reads its directory once.
    import torch_neuronx.metrics as metrics

    metrics.set_enabled(True)  # the compilation counters read 0 until enabled
    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    app = TorchNeuronWanApplication(
        model_path=str(args.model_dir), parallel=parallel, dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width, "num_frames": args.num_frames},
        exec_mode=args.exec_mode, device="neuron",
        enable_text_encoder=False, enable_transformer_2=True,
    )
    second = {}
    load_second = app.transformer_2.load

    def counted_load(*load_args, **load_kwargs):
        second["compilations_before"] = total_compilations()
        second["neffs_before"] = parity._neff_count()
        load_second(*load_args, **load_kwargs)
        second["compilations_after"] = total_compilations()
        second["neffs_after"] = parity._neff_count()

    app.transformer_2.load = counted_load
    with track_fallbacks() as load_fallbacks:
        app.load()

    high, low = [], []  # DiT forward seconds per expert
    app.transformer_adapter.forward_hook = high.append
    app.transformer_2_adapter.hook_from = None
    app.transformer_2_adapter.forward_hook = low.append
    generator = torch.Generator().manual_seed(args.seed)
    prompt_embeds = torch.randn(1, app.text_seq_len, app.config.text_dim, generator=generator)
    with track_fallbacks() as fallbacks:
        out = app(
            prompt_embeds=prompt_embeds,
            generator=generator,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            output_type="latent",
        )
    latents = out.latents.float()
    evidence = parity.device_evidence(app, [*load_fallbacks, *fallbacks])

    new_compilations = second["compilations_after"] - second["compilations_before"]
    calls = app.expert_calls
    checks = {
        "second_expert_compiles_nothing": new_compilations == 0,
        "both_experts_ran": calls.get("transformer", 0) > 0 and calls.get("transformer_2", 0) > 0,
        "fallbacks": evidence["fallbacks"] == [],
        "unwarmed_shapes": evidence["unwarmed_shapes"] == [],
        "latents_finite": bool(torch.isfinite(latents).all()),
    }

    def summary(seconds):
        return {"forwards": len(seconds),
                "median_s": statistics.median(seconds) if seconds else None}

    result = {
        "rank": rank, "world_size": world, "ok": all(checks.values()), "checks": checks,
        "expert_calls": calls,
        "second_expert_load": {**second, "new_compilations": new_compilations,
                               "new_neffs": second["neffs_after"] - second["neffs_before"]},
        "forward_seconds": {"transformer": summary(high), "transformer_2": summary(low)},
        "peak_device_mem_gb": evidence["peak_device_mem_gb"],
        "phase_seconds": evidence["phase_seconds"],
        "compiled_blocks": len(evidence["compiled_blocks"]),
        "counters": evidence["counters"],
        "latents_sha256": hashlib.sha256(latents.contiguous().numpy().tobytes()).hexdigest(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "build": build_info(),
    }
    (args.out_dir / f"result-rank{rank}.json").write_text(json.dumps(result, indent=2))
    peak = result["peak_device_mem_gb"]
    print(f"[rank {rank}] {'PASS' if result['ok'] else 'FAIL'} checks={checks} "
          f"expert_calls={calls} new_compilations={new_compilations} "
          f"peak_device_mem_gb={peak if peak is None else round(peak, 2)} "
          f"forward median s: high={result['forward_seconds']['transformer']['median_s']} "
          f"low={result['forward_seconds']['transformer_2']['median_s']}", flush=True)

    ok = result["ok"]
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            shas = [json.loads((args.out_dir / f"result-rank{r}.json").read_text())["latents_sha256"]
                    for r in range(world)]
            identical = len(set(shas)) == 1
            ok = ok and identical
            print(f"[rank 0] latents identical on all {world} ranks: {identical}", flush=True)
        dist.destroy_process_group()
    if rank == 0:
        torch.save(latents, args.out_dir / "latents.pt")
        print(f"build: {result['build']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
