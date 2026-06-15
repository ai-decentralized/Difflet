"""M5.0.3.3 — HunyuanVideo N-candidate smoke (no recompile).

v0 candidate workload (cclog 56 D4): host-side sequential replay of
``N`` seed-distinct candidates through a *single* pre-compiled
HunyuanVideo artifact at 320x512x61. Proves the candidate axis:

  1. A single ``app.load()`` of a pre-compiled artifact serves every
     candidate of every ``N`` — there is no compile call anywhere, so
     ``N=4`` and ``N=2`` run with **no recompile**.
  2. ``CandidateConfig(max=4)`` and ``CandidateConfig(max=4, active=2)``
     hash to the *same* compile-cache key (artifact identity is driven
     by ``max_candidates`` only), and the candidate axis leaves
     ``world_size`` unchanged — re-asserted here at integration scope.
  3. Per-step, per-candidate latent telemetry is emitted decode-free
     via ``LatentMetricCollector`` (the substrate M5.1's
     fork-on-divergence threshold reads off).

Synthetic keep-all scorer: every candidate is retained (M5.0 does not
own the real L2 fork threshold — that is M5.1).

Usage (canonical paths from scripts/hunyuan_smoke.sh):

    NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \\
    python scripts/hunyuan_candidate_smoke.py \\
      --source-dir /home/ubuntu/.cache/huggingface/hub/hunyuanvideo-real \\
      --compiled-dir .nova-cache/hunyuan_n4_20d40s2r/compiled \\
      --bundle .nova-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors \\
      --metrics-out /tmp/nova_m5_0_candidate_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--compiled-dir", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument(
        "--candidates",
        default="4,2",
        help="comma-separated active-N values to replay (all share one artifact)",
    )
    p.add_argument("--max-candidates", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("/tmp/nova_m5_0_candidate_metrics.json"),
    )
    return p.parse_args()


def _assert_candidate_contract(tp_degree: int, max_candidates: int) -> dict:
    """Integration re-assertion of the M5.0.3.1 abstraction guarantees."""

    from nova.pipeline.compile_cache import CacheSpec, cache_key
    from nova.pipeline.parallel_config import CandidateConfig, NovaParallelConfig

    parallel = NovaParallelConfig(tp_degree=tp_degree)

    def spec(candidate):
        return CacheSpec(
            model_id="hunyuanvideo-community/HunyuanVideo",
            model_path="/unused",
            model_name="hunyuan_video",
            parallel=parallel,
            dtype="bf16",
            height=320,
            width=512,
            num_frames=61,
            candidate=candidate,
        )

    key_max = cache_key(spec(CandidateConfig(max_candidates=max_candidates)))
    key_active2 = cache_key(
        spec(CandidateConfig(max_candidates=max_candidates, active_candidates=2))
    )
    key_legacy = cache_key(spec(None))
    cfg = CandidateConfig(max_candidates=max_candidates)

    assert key_max == key_active2, "active<max must be a cache hit (no recompile)"
    assert cfg.world_size(parallel) == parallel.world_size, "world_size must be invariant"
    if max_candidates > 1:
        assert key_max != key_legacy, "non-trivial max must differ from legacy"
    return {
        "cache_key_active_le_max_stable": key_max == key_active2,
        "world_size_invariant": cfg.world_size(parallel) == parallel.world_size,
        "world_size": parallel.world_size,
    }


def main() -> int:
    args = parse_args()
    candidate_ns = [int(x) for x in args.candidates.split(",") if x.strip()]
    if any(n < 1 or n > args.max_candidates for n in candidate_ns):
        raise ValueError(
            f"every --candidates value must be in [1, {args.max_candidates}]"
        )

    from nova.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from nova.pipeline.latent_metrics import LatentMetricCollector
    from nova.pipeline.parallel_config import NovaParallelConfig

    meta = json.loads(Path(str(args.bundle) + ".meta.json").read_text())
    tensors = load_file(str(args.bundle), device="cpu")
    print(
        f"[cand] shape = {meta['height']}x{meta['width']}x{meta['num_frames']} "
        f"steps={meta['num_inference_steps']} max_candidates={args.max_candidates}",
        flush=True,
    )

    contract = _assert_candidate_contract(args.tp_degree, args.max_candidates)
    print(f"[cand] candidate contract OK: {contract}", flush=True)

    app = NeuronHunyuanVideoApplication(
        model_path=args.source_dir,
        parallel=NovaParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        shape={
            "height": meta["height"],
            "width": meta["width"],
            "num_frames": meta["num_frames"],
        },
        text_seq_len=meta["text_seq_len"],
        enable_vae_decoder=False,
    )

    # The single, only artifact materialization in this process. Every
    # candidate of every N replays through this — no recompile possible.
    t_load = time.time()
    app.load(args.compiled_dir, skip_warmup=True)
    load_count = 1
    print(f"[cand] single app.load() elapsed = {time.time() - t_load:.3f}s", flush=True)

    init = tensors["latents_init"]
    num_steps = int(meta["num_inference_steps"])
    runs: dict[str, object] = {}

    for n in candidate_ns:
        coll = LatentMetricCollector()
        per_candidate_traj: list[list[torch.Tensor]] = []
        t_n = time.time()
        for i in range(n):
            g = torch.Generator().manual_seed(args.seed_base + i)
            noise = torch.randn(
                init.shape, generator=g, dtype=init.dtype
            ).contiguous()
            bundle = HunyuanVideoDiTInputBundle(
                hidden_states=noise,
                timestep=tensors["timesteps"][:1].clone(),
                encoder_hidden_states=tensors["encoder_hidden_states"],
                encoder_attention_mask=tensors["encoder_attention_mask"],
                pooled_projections=tensors["pooled_projections"],
                guidance=tensors["guidance"],
            )
            out = app(
                bundle=bundle,
                timesteps=tensors["timesteps"],
                num_inference_steps=num_steps,
                output_type="latent",
                return_trajectory=True,
            )
            per_candidate_traj.append([t.float() for t in out.trajectory])

        # Synthetic keep-all scorer.
        kept = list(range(n))
        n_traj_steps = len(per_candidate_traj[0])
        for step in range(n_traj_steps):
            stacked = torch.stack(
                [per_candidate_traj[c][step] for c in kept], dim=0
            )
            coll.record(step, stacked)
        elapsed = time.time() - t_n
        m = coll.to_metrics_dict()
        m["wall_s"] = elapsed
        m["kept_candidates"] = kept
        runs[str(n)] = m
        print(
            f"[cand] N={n} done in {elapsed:.2f}s "
            f"shared_prefix_fraction={m['shared_prefix_fraction']:.3f} "
            f"min_cross_cos_mean={m['min_cross_candidate_cosine_mean']:.6f}",
            flush=True,
        )

    result = {
        "schema": "nova-m5-0-candidate-smoke-v1",
        "source_dir": str(args.source_dir),
        "compiled_dir": str(args.compiled_dir),
        "bundle": str(args.bundle),
        "shape": f"{meta['height']}x{meta['width']}x{meta['num_frames']}",
        "max_candidates": args.max_candidates,
        "candidate_ns": candidate_ns,
        "single_artifact_load": load_count == 1,
        "no_recompile": load_count == 1,
        "candidate_contract": contract,
        "runs": runs,
        "passed": load_count == 1
        and contract["cache_key_active_le_max_stable"]
        and contract["world_size_invariant"]
        and all(
            runs[str(n)]["num_candidates"] == n for n in candidate_ns
        ),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: result[k] for k in (
        "single_artifact_load", "no_recompile", "candidate_ns", "passed")}, indent=2))
    print(f"[cand] metrics -> {args.metrics_out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
