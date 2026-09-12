"""Benchmark runner CLI — backend-generic, one model at a time.

    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python -m benchmark.bench --model ltx_2                 # one model
    python -m benchmark.bench --all                          # every model in the matrix
    python -m benchmark.bench --model flux_1_dev --skip-download

    DIFFLET_BENCH_DEVICE=v5e DIFFLET_BACKEND=tpu \\
        python -m benchmark.bench --backend tpu --model wan_2_1 --skip-download

For each model it runs download -> compile -> generate via the selected backend
adapter, collects the universal metrics, writes ``benchmark/<device>/<slug>.json``
and ``benchmark/<device>/<slug>.md`` (the detailed report).

The protocol is the one the trn2 folder was measured with, in this order:

1. **e2e cold** -- OS page cache dropped, then ONE fresh process:
   weights load + one generate to a decoded output.
2. **e2e warm** -- ``--warm-discard`` fresh processes thrown away (cache
   warming), then ``--iters`` fresh processes reported (n=3 on trn2).
3. **resident request** -- adapters that keep the model loaded
   (``supports_resident_mode``) serve ``--resident-iters`` more requests on the
   last process: the served-request cost, and the source of the per-step
   figure (device-synced real-loop deltas, step 0 excluded, all warm).
4. **natural** -- ``--natural-iters`` resident requests with no per-step sync.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from benchmark import report
from benchmark.harness import BenchResult, Stats, drop_page_cache
from benchmark.models import MATRIX


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _make_adapter(name: str, save_dir: str | None = None):
    if name == "trainium":
        from benchmark.adapters.trainium import TrainiumAdapter
        return TrainiumAdapter()
    if name in ("diffusers", "cpu", "cuda"):
        from benchmark.adapters.diffusers_ref import DiffusersRefAdapter
        return DiffusersRefAdapter(device="cuda" if name == "cuda" else "cpu")
    if name == "tpu":
        from benchmark.adapters.tpu import TpuAdapter
        return TpuAdapter(save_dir=save_dir)
    raise SystemExit(f"unknown backend: {name}")


_TPU_NOTE = (
    "The config_label comes from models.py::MATRIX and describes the TRAINIUM recipe -- "
    "'attention_cte' is a Neuron kernel and is NOT what ran here; the TPU backend uses "
    "the Pallas fused attention kernel (torch_xla.experimental.custom_kernel) above 32M "
    "score elements and scaled_dot_product_attention below it. The tp=4/cp=1 sharding IS "
    "accurate: the DiT is split across 4 v5e chips, one worker process per chip. "
    "compile_seconds=0 means no AOT artifact was built or reused, not that compilation is "
    "free -- XLA compiles on each process's first execution, inside e2e cold AND e2e warm "
    "(both are fresh processes, as on trn2), and torch_xla cannot persist the executables. "
    "The resident-process request row is the same request with the weights already on the "
    "chips: what `difflet serve` delivers. Per-step is the device-synced real-loop rule "
    "(harness.RealLoopStepTimer), taken from the resident synced iterations."
)


def _cold_note(wall: float, dropped: bool, drop_requested: bool) -> str:
    if dropped:
        how = ("TRUE cold start (OS page cache dropped before the run), so the weight "
               "load is a real cold disk read.")
    elif drop_requested:
        how = "cache NOT dropped (sudo failed) -- not a guaranteed cold start."
    else:
        how = "cache not dropped (--no-drop-caches) -- not a guaranteed cold start."
    return f"e2e_cold = {wall:.0f} s -- {how}"


def _warm_note(st: Stats, discarded: int) -> str:
    return (f"e2e_warm = {st.mean:.0f} s (n={st.n}; reported after {discarded} discarded "
            "cache-warming run(s) so the OS page cache is warm). Every run is a fresh "
            "process that reloads the weights, so 'warm' = warm disk cache -> faster load, "
            "not a resident model.")


def run_one(slug: str, backend: str, *, skip_download: bool, skip_compile: bool,
            iters: int, natural_iters: int = 2, warm_discard: int = 1,
            resident_iters: int = 2, drop_caches: bool = True,
            save_dir: str | None = None) -> BenchResult:
    cfg = MATRIX[slug]
    adapter = _make_adapter(backend, save_dir=save_dir)
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
        res.notes.append(_TPU_NOTE)
    resident = bool(getattr(adapter, "supports_resident_mode", False))
    natural = bool(getattr(adapter, "supports_natural_mode", False))
    res.protocol = {
        "drop_page_cache_before_cold": bool(drop_caches),
        "page_cache_dropped": None,
        "warm_discarded_runs": int(warm_discard) if iters > 0 else 0,
        "warm_iters": int(iters),
        "resident_iters": int(resident_iters) if resident else 0,
        "natural_iters": int(natural_iters) if (resident and natural) else 0,
        "fresh_process_per_run": True,
    }
    try:
        if not skip_download:
            adapter.prepare(cfg)
        if not skip_compile:
            res.compile_seconds, res.compile_breakdown = adapter.compile(cfg)

        # ---- 1. cold: page cache dropped, one fresh process ----------------
        dropped = drop_page_cache() if drop_caches else False
        res.protocol["page_cache_dropped"] = dropped
        adapter.tag(f"{slug}_cold")
        g = adapter.run_generate(cfg)
        res.e2e_cold_seconds = g.get("wall_seconds")
        res.load_seconds = g.get("load_seconds")
        res.e2e_breakdown = g.get("e2e_breakdown")
        res.peak_device_mem_gb = g.get("peak_mem_gb")
        res.output = g.get("output")
        res.step_basis = g.get("step_basis") or ""
        res.notes.append(_cold_note(g["wall_seconds"], dropped, drop_caches))
        # Fallback per-step source (trn2 / diffusers: the generate log). The
        # resident synced iterations below replace it where they exist.
        cold_steps = list(g.get("step_seconds") or [])
        if cold_steps:
            res.step_latency = Stats.from_samples(cold_steps).__dict__
        last: dict | None = g
        last_natural: dict | None = None

        # ---- 2. warm: fresh processes, page cache warm ----------------------
        if iters > 0:
            for i in range(max(0, warm_discard)):
                adapter.tag(f"{slug}_warmup{i}")
                adapter.run_generate(cfg)
            warm = []
            for i in range(iters):
                adapter.tag(f"{slug}_warm{i}")
                last = adapter.run_generate(cfg)
                warm.append(last["wall_seconds"])
            st = Stats.from_samples(warm)
            res.e2e_warm = st.__dict__
            res.e2e_warm_breakdown = last.get("e2e_breakdown")
            res.output = last.get("output") or res.output
            res.peak_device_mem_gb = last.get("peak_mem_gb") or res.peak_device_mem_gb
            res.notes.append(_warm_note(st, max(0, warm_discard)))

        # ---- 3. resident requests on the last process -----------------------
        if resident and resident_iters > 0:
            walls, steps = [], []
            for i in range(resident_iters):
                adapter.tag(f"{slug}_resident{i}")
                last = adapter.run_request(cfg, sync_steps=True)
                walls.append(last["wall_seconds"])
                steps.extend(last.get("step_seconds") or [])
            res.e2e_warm_resident = Stats.from_samples(walls).__dict__
            res.step_basis = last.get("step_basis") or res.step_basis
            if steps:
                # Per-step from warm, resident iterations only: a cold
                # process's deltas can still carry compiles past step 0.
                res.step_latency = Stats.from_samples(steps).__dict__
            # ---- 4. natural: same request, no per-step sync ---------------
            if natural and natural_iters > 0:
                nat_wall, nat_step = [], []
                for i in range(natural_iters):
                    adapter.tag(f"{slug}_natural{i}")
                    last_natural = adapter.run_request(cfg, sync_steps=False)
                    nat_wall.append(last_natural["wall_seconds"])
                    if last_natural.get("throughput_step_seconds") is not None:
                        nat_step.append(last_natural["throughput_step_seconds"])
                res.e2e_warm_natural = Stats.from_samples(nat_wall).__dict__
                if nat_step:
                    res.step_latency_natural = Stats.from_samples(nat_step).__dict__

        if res.step_latency and res.step_latency.get("mean", 0) > 0:
            res.throughput["steps/s"] = 1.0 / res.step_latency["mean"]
        if last:
            # Stage split, TeaCache stats and the throughput basis come from
            # the last WARM synced generate/request, never the cold one:
            # taking them from the cold generate folds XLA's first-execution
            # compile into them (it read as a 2.08 s/step "throughput"
            # against a real 0.5 s).
            res.stage_seconds = last.get("stage_seconds") or {}
            res.teacache = last.get("teacache")
            alt: dict = {}
            if last.get("throughput_step_seconds") is not None:
                alt["throughput"] = last["throughput_step_seconds"]
            # The enqueue rate is only meaningful when the deltas were NOT
            # synced; with a sync they are the same samples as step_latency.
            if last_natural and last_natural.get("enqueue_step_seconds"):
                samples = last_natural["enqueue_step_seconds"]
                alt["enqueue_mean"] = sum(samples) / len(samples)
            res.step_latency_alt = alt
        res.status = "ok"
    except Exception as e:  # keep partial results + record the failure honestly
        res.status = "failed"
        res.notes.append(f"FAILED: {e}")
    finally:
        try:
            adapter.shutdown()
        except Exception:  # noqa: BLE001
            pass
    return res


def write_outputs(slug: str, res: BenchResult) -> None:
    from benchmark.models import DEVICE, results_dir, json_path, report_path
    Path(results_dir()).mkdir(parents=True, exist_ok=True)
    d = res.to_dict()
    d["config_slug"] = slug
    d["device_slug"] = DEVICE
    # carry the hardware-agnostic repro fields into the JSON so the report's
    # Reproduction section is complete and matches the schema across devices.
    cfg = MATRIX[slug]
    d["revision"] = cfg.revision
    d["seed"] = cfg.seed
    d["prompt"] = cfg.prompt
    d["guidance_scale"] = cfg.guidance_scale
    d["output_kind"] = cfg.output_kind
    Path(json_path(slug)).write_text(json.dumps(d, indent=2))
    Path(report_path(slug)).write_text(report.render(d))
    print(f"[bench] {slug}: status={res.status}  "
          f"compile={res.compile_seconds}  e2e_cold={res.e2e_cold_seconds}  "
          f"-> {report_path(slug)}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Difflet diffusion benchmark")
    p.add_argument("--model", help="matrix slug (e.g. ltx_2, flux_1_dev)")
    p.add_argument("--all", action="store_true", help="run every model in the matrix")
    p.add_argument("--backend", default="trainium")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-compile", action="store_true",
                   help="reuse an existing compile cache")
    p.add_argument("--iters", type=int, default=0,
                   help="warm end-to-end iterations, each a fresh process with the page "
                        "cache warm (cold run always done; trn2 used 3)")
    p.add_argument("--warm-discard", type=int, default=1,
                   help="fresh-process runs discarded before the warm iterations, so the "
                        "page cache is warm (trn2 used 1, 2 for the biggest checkpoints)")
    p.add_argument("--resident-iters", type=int, default=2,
                   help="extra requests on the resident process after the warm runs "
                        "(adapters advertising supports_resident_mode); source of the "
                        "per-step figure and the served-request row")
    p.add_argument("--natural-iters", type=int, default=2,
                   help="resident requests run WITHOUT the per-step device sync, "
                        "reported as the natural basis (0 disables)")
    p.add_argument("--no-drop-caches", dest="drop_caches", action="store_false",
                   help="skip `sync; echo 3 > /proc/sys/vm/drop_caches` before the cold run")
    p.add_argument("--save-dir", default=None,
                   help="directory for the decoded outputs of every run (png / mp4 + "
                        "contact sheet), named <slug>_<phase>")
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
        print(f"\n========== benchmarking {slug} ({MATRIX[slug].model_id}) ==========", flush=True)
        res = run_one(slug, args.backend, skip_download=args.skip_download,
                      skip_compile=args.skip_compile, iters=args.iters,
                      natural_iters=args.natural_iters, warm_discard=args.warm_discard,
                      resident_iters=args.resident_iters, drop_caches=args.drop_caches,
                      save_dir=args.save_dir)
        write_outputs(slug, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
