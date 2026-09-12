"""Benchmark runner CLI — backend-generic, one model at a time.

    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python -m benchmark.bench --model ltx_2                 # one model
    python -m benchmark.bench --all                          # every model in the matrix
    python -m benchmark.bench --model flux_1_dev --skip-download

For each model it runs download -> compile -> generate via the selected backend
adapter, collects the universal metrics, writes ``benchmark/<device>/<slug>.json``
and ``benchmark/<slug>.md`` (the detailed report).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from benchmark import report
from benchmark.harness import BenchResult, Stats
from benchmark.models import MATRIX, add_config_arg, resolve


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _make_adapter(name: str):
    if name == "trainium":
        from benchmark.adapters.trainium import TrainiumAdapter
        return TrainiumAdapter()
    if name in ("diffusers", "cpu", "cuda"):
        from benchmark.adapters.diffusers_ref import DiffusersRefAdapter
        return DiffusersRefAdapter(device="cuda" if name == "cuda" else "cpu")
    if name == "tpu":
        from benchmark.adapters.tpu import TpuAdapter
        return TpuAdapter()
    raise SystemExit(f"unknown backend: {name}")


def run_one(slug: str, backend: str, *, skip_download: bool, skip_compile: bool,
            iters: int, natural_iters: int = 2, config: str = "tp4") -> BenchResult:
    cfg = resolve(slug, config)
    adapter = _make_adapter(backend)
    res = BenchResult(
        model_id=cfg.model_id, model_type=cfg.model_type, backend=backend,
        device=adapter.device_info(), dtype=cfg.dtype,
        parallel=cfg.parallel_dict(),
        shape={"height": cfg.height, "width": cfg.width, "num_frames": cfg.num_frames},
        steps=cfg.steps, toolchain=adapter.toolchain(),
        config_label=cfg.config_label, timestamp=_utc(),
    )
    res.notes.append(cfg.notes) if cfg.notes else None
    if backend in ("diffusers", "cuda", "cpu"):
        res.notes.append(
            "H100/CUDA reference runs single-GPU DENSE via stock diffusers (eager, no AOT "
            "compile). The tp=4/cp=1 shown in Configuration/Reproduction is the Trainium "
            "sharding for the difflet recipe — NOT how this GPU run executed (effective "
            "tp=1). Per-step latency is the load-independent metric comparable to trn2.")
    if backend == "tpu":
        res.notes.append(
            "The config_label comes from models.py::MATRIX and describes the "
            "TRAINIUM recipe — 'attention_cte' is a Neuron kernel and is NOT what "
            "ran here; the TPU backend uses scaled_dot_product_attention. The "
            "tp=4/cp=1 sharding IS accurate: the DiT is split across 4 v5e chips, "
            "one worker process per chip. Attention uses the Pallas fused kernel "
            "(torch_xla.experimental.custom_kernel) above 32M score elements and "
            "scaled_dot_product_attention below it. compile_seconds=0 means no AOT "
            "artifact was built or reused, not that compilation is free — XLA "
            "compiles on each process's first execution, inside e2e_cold. "
            "step_latency is on a throughput basis (denoise wall clock / steps): "
            "unsynced inter-step deltas under lazy XLA measure the enqueue rate, "
            "not device time.")
    try:
        if not skip_download:
            adapter.prepare(cfg)
        if not skip_compile:
            res.compile_seconds, res.compile_breakdown = adapter.compile(cfg)
        # cold generate
        g = adapter.run_generate(cfg)
        res.e2e_cold_seconds = g.get("wall_seconds")
        res.load_seconds = g.get("load_seconds")
        res.e2e_breakdown = g.get("e2e_breakdown")
        if g.get("step_seconds"):
            st = Stats.from_samples(g["step_seconds"])
            res.step_latency = st.__dict__
            if st.mean > 0:
                res.throughput["steps/s"] = 1.0 / st.mean
        res.peak_device_mem_gb = g.get("peak_mem_gb")
        res.output = g.get("output")
        # Carried through so a backend can publish the comparable number and
        # still show what its own basis costs it. stage_seconds was previously
        # computed by the adapters and then dropped on the floor here.
        res.step_basis = g.get("step_basis") or ""
        # NOTE on per-step: the marginal-across-process method (run at 2 step
        # counts, subtract) is NOT used — each `difflet generate` reloads the
        # text encoder (5-11 GB) whose latency swings with OS page-cache warmth,
        # so the "fixed" overhead does not cancel (it produced nonsense, even
        # negative, per-step). The stable per-step is the in-process warm DiT
        # forward measured by the per-model *_transformer_parity.py harness
        # (prints "trainium forward elapsed"); see each report.
        if iters > 0:
            warm = []
            last = None
            for _ in range(iters):
                last = adapter.run_generate(cfg)
                warm.append(last["wall_seconds"])
            res.e2e_warm = Stats.from_samples(warm).__dict__
            # Stage split and the alternative per-step bases come from a WARM
            # iteration. Taking them from the cold generate folds XLA's
            # first-execution compile into them -- it read as a 2.08 s/step
            # "throughput" against a real 0.5 s.
            if natural_iters > 0 and getattr(adapter, "supports_natural_mode", False):
                # Same generate, no per-step sync: what a real serving loop
                # actually delivers, as opposed to what the cross-device rule
                # measures. Both are kept; neither replaces the other.
                nat_wall, nat_step = [], []
                for _ in range(max(1, natural_iters)):
                    n = adapter.run_generate(cfg, sync_steps=False)
                    nat_wall.append(n["wall_seconds"])
                    if n.get("throughput_step_seconds") is not None:
                        nat_step.append(n["throughput_step_seconds"])
                res.e2e_warm_natural = Stats.from_samples(nat_wall).__dict__
                if nat_step:
                    res.step_latency_natural = Stats.from_samples(nat_step).__dict__
            if last:
                res.stage_seconds = last.get("stage_seconds") or {}
                alt: dict = {}
                if last.get("throughput_step_seconds") is not None:
                    alt["throughput"] = last["throughput_step_seconds"]
                # Only meaningful when the deltas were NOT synced; with a sync
                # they are the same samples as step_latency, and reporting them
                # as a second basis would invent a distinction that is not there.
                if last.get("step_basis") == "throughput" and last.get("enqueue_step_seconds"):
                    samples = last["enqueue_step_seconds"]
                    alt["enqueue_mean"] = sum(samples) / len(samples)
                res.step_latency_alt = alt
        res.status = "ok"
    except Exception as e:  # keep partial results + record the failure honestly
        res.status = "failed"
        res.notes.append(f"FAILED: {e}")
    return res


def write_outputs(slug: str, res: BenchResult, config: str = "tp4") -> None:
    from benchmark.models import DEVICE, results_dir, json_path, report_path
    Path(results_dir()).mkdir(parents=True, exist_ok=True)
    d = res.to_dict()
    cfg = resolve(slug, config)
    d["config_slug"] = cfg.config_slug   # result-file stem (<slug>[_<config>])
    d["model_slug"] = slug               # MATRIX key, for the harness repro commands
    d["config"] = config                 # parallel-topology label
    d["device_slug"] = DEVICE
    # carry the hardware-agnostic repro fields into the JSON so the report's
    # Reproduction section is complete and matches the schema across devices.
    d["revision"] = cfg.revision
    d["seed"] = cfg.seed
    d["prompt"] = cfg.prompt
    d["guidance_scale"] = cfg.guidance_scale
    d["output_kind"] = cfg.output_kind
    Path(json_path(cfg.config_slug)).write_text(json.dumps(d, indent=2))
    Path(report_path(cfg.config_slug)).write_text(report.render(d))
    print(f"[bench] {cfg.config_slug}: status={res.status}  "
          f"compile={res.compile_seconds}  e2e_cold={res.e2e_cold_seconds}  "
          f"-> {report_path(cfg.config_slug)}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Difflet diffusion benchmark")
    p.add_argument("--model", help="matrix slug (e.g. ltx_2, flux_1_dev)")
    p.add_argument("--all", action="store_true", help="run every model in the matrix")
    p.add_argument("--backend", default="trainium")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-compile", action="store_true",
                   help="reuse an existing compile cache")
    p.add_argument("--natural-iters", type=int, default=2,
                   help="extra warm iterations run WITHOUT the per-step device "
                        "sync, reported as the natural basis (0 disables). Only "
                        "used by adapters advertising supports_natural_mode.")
    p.add_argument("--iters", type=int, default=0,
                   help="extra warm end-to-end iterations (cold run always done)")
    add_config_arg(p)
    args = p.parse_args()

    if args.all:
        slugs = list(MATRIX)
    elif args.model:
        if args.model not in MATRIX:
            raise SystemExit(f"unknown model '{args.model}'. known: {', '.join(MATRIX)}")
        slugs = [args.model]
    else:
        raise SystemExit("specify --model <slug> or --all")

    for slug in slugs:
        print(f"\n========== benchmarking {slug} ({MATRIX[slug].model_id}) "
              f"[{args.config}] ==========", flush=True)
        res = run_one(slug, args.backend, skip_download=args.skip_download,
                      skip_compile=args.skip_compile, iters=args.iters,
                      natural_iters=args.natural_iters, config=args.config)
        write_outputs(slug, res, args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
