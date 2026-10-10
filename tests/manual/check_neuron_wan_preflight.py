"""Manual check (W1b): Wan2.1-T2V-1.3B through the neuron Wan DiT application on 4 ranks.

The pre-flight of the 14B gates: the identical code path (``NeuronWanTransformerApplication``,
TP4, the one static shape 480x832x9 = latent (1,16,3,60,104) = 4,680 DiT tokens, 512 text
tokens, the shared block graph in compile mode) on the 1.3B checkpoint (30 blocks, dim 1536,
12 heads -> 3 per rank), so the cold block-graph compile, per-rank HBM, the one-graph property
and parity with the diffusers fp32 reference are measured in minutes instead of hours.

Inputs: the diffusers snapshot (``--model-dir``, holding ``transformer/``) and the CPU reference
written beforehand by ``tests/numerical/tpu_oracle_reference.py wan --timesteps 996,500
--snapshot <model-dir> --out <ref>`` (fp32 forward + bf16 control per timestep, same seeded
latents and text). Run under torchrun with 4 ranks, one device job at a time, e.g. (absolute
paths, ``W1B`` a records directory):

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3 DIFFLET_NO_JEMALLOC=1 DIFFLET_DISABLE_PREWARM=1 \
    PYTHONPATH=<worktree> timeout 1800 <venv>/bin/torchrun --standalone --nproc-per-node 4 \
        <worktree>/tests/manual/check_neuron_wan_preflight.py --model-dir <snapshot> \
        --ref W1B/oracle_1p3b.pt --exec-mode eager --timesteps 996,500 \
        --cache-dir W1B/eager_cache --out-dir W1B/run-eager
    rm -rf W1B/compile_cache W1B/compile_cache_local W1B/compile_cache.cold.json   # cold
    DIFFLET_NEURON_PG_TIMEOUT_S=3600 ... --exec-mode compile --cache-dir W1B/compile_cache \
        --expect cold --eager-dir W1B/run-eager --out-dir W1B/run-cold
    ... --exec-mode compile --cache-dir W1B/compile_cache --expect warm \
        --eager-dir W1B/run-eager --cold-dir W1B/run-cold --out-dir W1B/run-warm

``--cache-dir`` follows tests/manual/check_neuron_compile_c9.py: it is the NEFF cache, its
``_local`` sibling holds the compile locks and ``<dir>.cold.json`` the cold run's warm-up time.

Each rank enables ``torch_neuronx.metrics`` before any device work (otherwise the compilation
counters read 0 and "warm" proves nothing), loads with ``load(skip_warmup=True)`` and then runs
``warmup()`` (the two halves of ``load()``, split only to read the counters in between), runs
one DiT forward per timestep through the composed application's host<->device adapter under
``track_fallbacks()``, and writes ``<out-dir>/result-rank<r>.json``: the checks, the parity
metrics, ``parity.device_evidence``, ``max_memory_allocated``, ``phase_seconds``, the compilation
counters / ``CompilationCache.CompilationTime`` / Dynamo graph counts / NEFF count / device and
host memory at start, after load, after the warm-up and at the end, the driver's per-core device
memory counters, the process-group timeout, and the sha256 of its outputs. Rank 0 also writes
its outputs as fp32 on the host, ``<out-dir>/out-t<timestep>.pt``.

Checks (on every rank; rank 0 prints one PASS/FAIL line per check, each rank ``[rank r] PASS``):

* checkpoint: every parameter found, no unexpected tensor; heads_per_rank: 12 / 4 = 3;
* fallbacks: ``track_fallbacks()`` around the warm-up and around the forwards is ``[]``;
  unwarmed_shapes: ``[]``;
* parity_fp32_t<ts>: ``parity.check`` against the fp32 reference with the bf16 control, at the
  TPU thresholds (cosine >= 0.995, rel_l1 <= 1.15x the control's, same-dtype <= 1.2x);
* compile mode: compiled_blocks (all 30), one_graph (Dynamo ``unique_graphs == 1`` after the
  warm-up and after the forwards, no graph break), warmup_neff (the warm-up compiled or loaded a
  NEFF), no_recompile_in_forwards (no new Dynamo graph after the warm-up and, unless the cache
  was cold, no compilation during the forwards; the cache entries each phase added are in the
  JSON);
* ``--eager-dir``: vs_eager_t<ts>, the compiled output within ``rel_max <= 2e-2`` of the eager one;
* ``--expect cold``: rank 0 finds new ``.neff`` files and the lock directory; ``--expect warm``:
  no compilation, at least one persistent hit and no new ``.neff`` on every rank, and the
  warm-up under half the cold one; ``--cold-dir``: outputs ``torch.equal`` to the cold run's.

The process group is destroyed only when every check passed (a failing rank exits non-zero and
torchrun stops its peers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["DIFFLET_BACKEND"] = "neuron"

import torch

from difflet.backends.neuron import runtime
from difflet.backends.neuron.compile import NEFF_CACHE_ENV, configure_compile_cache
from tests.numerical import neuron_parity as parity
from difflet.backends.neuron.runtime import track_fallbacks
from difflet.backends.neuron.wan.config import NeuronWanConfig
from difflet.backends.neuron.wan.transformer import NeuronWanTransformerApplication
from difflet.backends.registry import get_backend
from difflet.models.wan.neuron_application import _OnDeviceTransformer
from difflet.pipeline.parallel_config import DiffletParallelConfig

COMPILE_VS_EAGER_MAX_REL = 2e-2
COMPILATION_TIME = "CompilationCache.CompilationTime"
SYSFS_DEVICE = Path("/sys/devices/virtual/neuron_device/neuron0")
GIB = 2**30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True,
                   help="diffusers snapshot directory holding transformer/")
    p.add_argument("--ref", type=Path, required=True,
                   help="CPU reference from tpu_oracle_reference.py wan --timesteps ...")
    p.add_argument("--exec-mode", choices=("eager", "compile"), required=True)
    p.add_argument("--timesteps", default="996,500",
                   help="comma-separated timesteps; each must be in the reference's runs")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="NEFF cache directory (else the Difflet compile cache's)")
    p.add_argument("--expect", choices=("cold", "warm"), default=None)
    p.add_argument("--eager-dir", type=Path, default=None,
                   help="an eager run's --out-dir: gate compiled vs eager outputs")
    p.add_argument("--cold-dir", type=Path, default=None,
                   help="a cold compile run's --out-dir: outputs must be torch.equal")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--text-seq-len", type=int, default=512)
    args = p.parse_args()
    if args.expect is not None and args.exec_mode != "compile":
        p.error("--expect applies to --exec-mode compile")
    if args.expect is not None and args.cache_dir is None:
        p.error("--expect needs --cache-dir (the directory whose cold/warm state is checked)")
    try:
        args.timesteps = [float(t) for t in args.timesteps.split(",")]
    except ValueError:
        p.error(f"--timesteps must be comma-separated numbers, got {args.timesteps!r}")
    return args


# ------------------------------------------------------------------------- measurements


def count_neffs() -> int:
    directory = Path(os.environ[NEFF_CACHE_ENV])
    return sum(1 for _ in directory.rglob("*.neff")) if directory.is_dir() else 0


def compilation_counters() -> dict[str, int]:
    import torch_neuronx.metrics as metrics

    return {name: metrics.get_counter_value("CompilationCache." + name) or 0
            for name in parity.COUNTER_NAMES}


def compilation_time() -> dict[str, Any]:
    """``CompilationCache.CompilationTime`` in seconds: the sum and count over every sample,
    and the largest retained samples (the circular history is bounded)."""
    import torch_neuronx.metrics as metrics

    data = metrics.get_metric_data(COMPILATION_TIME)
    if not data:
        return {"total_s": 0.0, "samples": 0, "largest_s": []}
    values = sorted((s["value"] / 1e9 for s in data["samples"]), reverse=True)
    return {"total_s": data["accumulator"] / 1e9, "samples": data["total_samples"],
            "retained": len(values), "largest_s": [round(v, 3) for v in values[:8]]}


def dynamo_stats() -> dict[str, int]:
    from torch._dynamo.utils import counters

    return {"unique_graphs": int(counters["stats"]["unique_graphs"]),
            "graph_breaks": int(sum(counters["graph_break"].values()))}


def device_memory() -> dict[str, float]:
    import torch_neuronx

    allocated, peak = torch_neuronx.memory_allocated(), torch_neuronx.max_memory_allocated()
    return {"allocated_bytes": allocated, "max_allocated_bytes": peak,
            "allocated_gib": allocated / GIB, "max_allocated_gib": peak / GIB}


def host_memory() -> dict[str, float]:
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmHWM:", "VmRSS:")):
            key, value = line.split(":", 1)
            values[f"{key}_gib"] = int(value.split()[0]) / 2**20  # kB
    return values


def cache_entries() -> dict[str, dict[str, Any]] | None:
    """The runtime's in-memory compilation cache (NEFFs compiled or loaded by this process),
    keyed by cache key; None if this torch-neuronx does not expose it."""
    import torch_neuronx

    try:
        entries = torch_neuronx._C._get_compilation_cache_entries()
    except AttributeError:
        return None
    return {e["cache_key"]: {"neff_size_bytes": e["neff_size_bytes"],
                             "compilation_time_ms": e["compilation_time_ms"]} for e in entries}


def snapshot(t0: float, *, device: bool = True) -> dict[str, Any]:
    return {
        "t_s": time.perf_counter() - t0,
        "counters": compilation_counters(),
        "compilation_time": compilation_time(),
        "dynamo": dynamo_stats(),
        "neff_count": count_neffs(),
        "device_memory": device_memory() if device else None,
        "host_memory": host_memory(),
        "cache_entries": cache_entries(),
    }


def added_entries(after: dict[str, Any], before: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The cache entries a phase added (largest NEFF first)."""
    if after["cache_entries"] is None or before["cache_entries"] is None:
        return None
    added = [{"key": k, **v} for k, v in after["cache_entries"].items()
             if k not in before["cache_entries"]]
    return sorted(added, key=lambda e: -(e["neff_size_bytes"] or 0))


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def sysfs_device_memory(logical_core: int | None) -> dict[str, Any]:
    """The Neuron driver's device-memory counters (bytes, present and peak per category) of the
    physical cores behind this rank's logical core. trn2 runs logical-neuroncore-config 2
    (neuron-ls), so logical core r spans physical cores 2r and 2r+1. Unlike
    ``max_memory_allocated`` (the torch allocator) these include the runtime's own use: NEFF
    code, scratchpad, collectives buffers."""
    if logical_core is None:
        return {}
    result: dict[str, Any] = {}
    for physical in (2 * logical_core, 2 * logical_core + 1):
        base = SYSFS_DEVICE / f"neuron_core{physical}" / "stats" / "memory_usage" / "device_mem"
        if not base.is_dir():
            continue
        entry: dict[str, Any] = {k: _read_int(base / k) for k in ("present", "peak")}
        for child in sorted(p for p in base.iterdir() if p.is_dir()):
            entry[child.name] = {k: _read_int(child / k) for k in ("present", "peak")}
        result[f"neuron_core{physical}"] = entry
    return result


def pg_timeout() -> dict[str, Any]:
    """The configured process-group timeout and, if the backend exposes it, the one it got."""
    import torch.distributed as dist

    configured = runtime.process_group_timeout()
    try:
        group = dist.distributed_c10d._get_default_group()
        backend = group._get_backend(torch.device("neuron"))
        seen: Any = backend.options.timeout.total_seconds()
    except Exception as exc:  # noqa: BLE001 - best effort, recorded as text
        seen = f"unreadable: {type(exc).__name__}: {exc}"
    return {"env": os.environ.get(runtime.PG_TIMEOUT_ENV),
            "configured_s": None if configured is None else configured.total_seconds(),
            "backend_options_timeout_s": seen}


def sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    return {k: after["counters"][k] - before["counters"][k] for k in after["counters"]}


# ------------------------------------------------------------------------------- run


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cache_dir = local_dir = timing_file = None
    if args.cache_dir is not None:
        cache_dir = args.cache_dir.expanduser().absolute()
        local_dir = cache_dir.with_name(cache_dir.name + "_local")
        timing_file = cache_dir.with_name(cache_dir.name + ".cold.json")
        # Before any device work: the runtime's compilation cache reads these once.
        configure_compile_cache(cache_dir, local_cache_dir=local_dir)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    prepare_s = time.perf_counter() - t0

    import torch.distributed as dist
    import torch_neuronx.metrics as metrics

    metrics.set_enabled(True)  # before any device work, or every counter reads 0 (D53)
    rank = dist.get_rank() if dist.is_initialized() else 0
    logical_core = runtime.parse_core_list(os.environ.get(runtime.VISIBLE_CORES_ENV))
    logical_core = logical_core[0] if len(logical_core) == 1 else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    snaps = {"start": snapshot(t0, device=False)}

    transformer_dir = args.model_dir / "transformer"
    config = NeuronWanConfig.from_pretrained(
        transformer_dir, tp_degree=world, height=args.height, width=args.width,
        num_frames=args.num_frames, text_seq_len=args.text_seq_len, torch_dtype=torch.bfloat16,
    )
    app = NeuronWanTransformerApplication(
        model_path=transformer_dir, config=config, parallel=parallel, dtype=torch.bfloat16,
        exec_mode=args.exec_mode, device="neuron",
    )
    app.load(skip_warmup=True)
    snaps["after_load"] = snapshot(t0)
    with track_fallbacks() as warmup_fallbacks:
        app.warmup()
    snaps["after_warmup"] = snapshot(t0)

    blob = torch.load(args.ref, weights_only=False)
    runs = {float(run["timestep"][0]): run for run in blob["runs"]}
    missing = [ts for ts in args.timesteps if ts not in runs]
    if missing:
        raise ValueError(f"{args.ref} has no reference at timesteps {missing} "
                         f"(it holds {sorted(runs)})")
    adapter = _OnDeviceTransformer(app, config, torch.bfloat16, app.device)
    forward_s: list[float] = []
    adapter.forward_hook = forward_s.append
    outputs: dict[float, torch.Tensor] = {}
    with track_fallbacks() as fallbacks:
        for ts in args.timesteps:
            out = adapter(blob["hidden_states"], runs[ts]["timestep"],
                          blob["encoder_hidden_states"])
            outputs[ts] = out.float()
    snaps["end"] = snapshot(t0)
    evidence = parity.device_evidence(app, fallbacks)

    # ------------------------------------------------------------------ checks
    checks: list[tuple[str, bool, str]] = []
    report = app.load_report or {}
    checks.append(("checkpoint", not report.get("missing") and not report.get("unexpected"),
                   f"missing={len(report.get('missing', []))} "
                   f"unexpected={len(report.get('unexpected', []))}"))
    attn = app.module.blocks[0].attn1
    heads_per_rank = config.num_attention_heads // world
    checks.append(("heads_per_rank", attn.local_heads == heads_per_rank,
                   f"local_heads={attn.local_heads} (={config.num_attention_heads}/{world})"))
    checks.append(("fallbacks", evidence["fallbacks"] == [] and warmup_fallbacks == [],
                   f"forwards={evidence['fallbacks']} warm-up={warmup_fallbacks}"))
    checks.append(("unwarmed_shapes", evidence["unwarmed_shapes"] == [],
                   f"{evidence['unwarmed_shapes']} warmup={app.warmup_shapes}"))

    parity_results: dict[str, Any] = {}
    for ts, out in outputs.items():
        ref, ref_bf16 = runs[ts]["reference"], runs[ts]["reference_bf16"]
        m = parity.metrics(out, ref)
        m["vs_bf16"] = parity.metrics(out, ref_bf16)
        control = parity.metrics(ref_bf16, ref)
        ok, reasons = parity.check(m, control=control)
        ratio = m["rel_l1"] / control["rel_l1"]
        same_ratio = m["vs_bf16"]["rel_l1"] / control["rel_l1"]
        parity_results[f"{ts:g}"] = {"vs_fp32": m, "control": control, "ok": ok,
                                    "reasons": reasons, "ratio_vs_control": ratio,
                                    "same_dtype_ratio": same_ratio}
        checks.append((f"parity_fp32_t{ts:g}", ok,
                       f"cos={m['cosine']:.6f} rel_l1={m['rel_l1']:.3e} "
                       f"({ratio:.3f}x control {control['rel_l1']:.3e}, cos "
                       f"{control['cosine']:.6f}) vs_bf16 rel_l1={m['vs_bf16']['rel_l1']:.3e} "
                       f"({same_ratio:.3f}x) rel_max={m['rel_max']:.3e}"
                       + (f" {reasons}" if reasons else "")))

    warmup_delta = delta(snaps["after_warmup"], snaps["after_load"])
    forwards_delta = delta(snaps["end"], snaps["after_warmup"])
    run_delta = delta(snaps["end"], snaps["start"])
    new_neffs = snaps["end"]["neff_count"] - snaps["start"]["neff_count"]
    graphs_warm = snaps["after_warmup"]["dynamo"]
    graphs_end = snaps["end"]["dynamo"]
    if args.exec_mode == "compile":
        checks.append(("compiled_blocks", len(app.compiled_blocks) == config.num_layers,
                       f"{len(app.compiled_blocks)} of {config.num_layers}"))
        checks.append(("one_graph",
                       graphs_warm["unique_graphs"] == 1 and graphs_end["unique_graphs"] == 1
                       and graphs_end["graph_breaks"] == 0,
                       f"unique_graphs after warm-up={graphs_warm['unique_graphs']} "
                       f"end={graphs_end['unique_graphs']} graph_breaks="
                       f"{graphs_end['graph_breaks']} for {config.num_layers} blocks"))
        checks.append(("warmup_neff",
                       warmup_delta["TotalCompilations"] + warmup_delta["PersistentHits"] >= 1,
                       f"warm-up counters {warmup_delta}"))
        # A new Dynamo graph after the warm-up is a recompile in every run. A NEFF compiled
        # during the forwards is one only on a warm cache: on a cold one a compile the warm-up
        # started can finish during the first forward (seen once in four cold runs).
        checks.append(("no_recompile_in_forwards",
                       graphs_end["unique_graphs"] == graphs_warm["unique_graphs"]
                       and (args.expect == "cold" or forwards_delta["TotalCompilations"] == 0),
                       f"unique_graphs {graphs_warm['unique_graphs']} -> "
                       f"{graphs_end['unique_graphs']}, forward counters {forwards_delta}"))
    else:
        checks.append(("eager_compiles_no_block", not app.compiled_blocks
                       and graphs_end["unique_graphs"] == 0,
                       f"compiled_blocks={len(app.compiled_blocks)} "
                       f"unique_graphs={graphs_end['unique_graphs']}"))

    if args.eager_dir is not None:
        for ts, out in outputs.items():
            eager = torch.load(args.eager_dir / f"out-t{ts:g}.pt", weights_only=True)
            m = parity.metrics(out, eager)
            parity_results[f"{ts:g}"]["vs_eager"] = m
            checks.append((f"vs_eager_t{ts:g}", m["rel_max"] <= COMPILE_VS_EAGER_MAX_REL,
                           f"rel_max={m['rel_max']:.3e} (tol {COMPILE_VS_EAGER_MAX_REL:.0e}) "
                           f"cos={m['cosine']:.6f} bit_identical={torch.equal(out, eager)}"))

    warmup_s = app.phase_seconds["warmup forward"]
    cold_warmup_s = None
    if args.expect == "cold" and rank == 0:
        checks.append(("neff_cache_cold", new_neffs >= 1 and local_dir.is_dir(),
                       f"new_neffs={new_neffs} lock_dir={local_dir.is_dir()} run={run_delta}"))
        timing_file.write_text(json.dumps({"warmup_forward_s": warmup_s}))
    if args.expect == "warm":
        checks.append(("warm_counters",
                       run_delta["TotalCompilations"] == 0 and run_delta["PersistentHits"] >= 1
                       and new_neffs == 0,
                       f"run counters {run_delta} new_neffs={new_neffs}"))
        if rank == 0:
            if timing_file.exists():
                cold_warmup_s = json.loads(timing_file.read_text())["warmup_forward_s"]
            checks.append(("warm_start_time",
                           cold_warmup_s is not None and warmup_s < 0.5 * cold_warmup_s,
                           f"warm={warmup_s:.1f}s cold="
                           + ("missing" if cold_warmup_s is None else f"{cold_warmup_s:.1f}s")))
    if args.cold_dir is not None:
        for ts, out in outputs.items():
            cold = torch.load(args.cold_dir / f"out-t{ts:g}.pt", weights_only=True)
            equal = torch.equal(out, cold)
            parity_results[f"{ts:g}"]["equal_to_cold"] = equal
            checks.append((f"equal_to_cold_t{ts:g}", equal,
                           f"max_abs={float((out - cold).abs().max()):.3e}"))

    ok = all(passed for _, passed, _ in checks)
    if rank == 0:
        for ts, out in outputs.items():
            torch.save(out, args.out_dir / f"out-t{ts:g}.pt")

    result = {
        "rank": rank, "world_size": world, "logical_core": logical_core,
        "exec_mode": args.exec_mode, "expect": args.expect, "ok": ok,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "neff_cache_dir": os.environ.get(NEFF_CACHE_ENV),
        "metrics_enabled": bool(metrics.is_enabled()),
        "pg_timeout": pg_timeout(),
        "env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "HF_HUB_OFFLINE", "DIFFLET_NO_JEMALLOC", runtime.PG_TIMEOUT_ENV,
            runtime.VISIBLE_CORES_ENV, "TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT")},
        "torch_threads": torch.get_num_threads(),
        "config": {"num_layers": config.num_layers, "dim": config.num_attention_heads
                   * config.attention_head_dim, "heads": config.num_attention_heads,
                   "heads_per_rank": heads_per_rank, "ffn_dim": config.ffn_dim,
                   "image_seq_len": config.image_seq_len, "text_seq_len": config.text_seq_len,
                   "latent": [config.latent_frames, config.latent_height, config.latent_width]},
        "checks": [{"name": n, "ok": p, "detail": d} for n, p, d in checks],
        "parity": parity_results,
        "device_evidence": evidence,
        "warmup_fallbacks": list(warmup_fallbacks),
        "max_memory_allocated_bytes": snaps["end"]["device_memory"]["max_allocated_bytes"],
        "max_memory_allocated_gib": snaps["end"]["device_memory"]["max_allocated_gib"],
        "weights_allocated_gib": snaps["after_load"]["device_memory"]["allocated_gib"],
        "phase_seconds": dict(app.phase_seconds),
        "timers_s": {"prepare_runtime": prepare_s, "forwards": forward_s,
                     "total": time.perf_counter() - t0},
        "counters": {"warmup": warmup_delta, "forwards": forwards_delta, "run": run_delta},
        "cache_entries_added": {
            "load": added_entries(snaps["after_load"], snaps["start"]),
            "warmup": added_entries(snaps["after_warmup"], snaps["after_load"]),
            "forwards": added_entries(snaps["end"], snaps["after_warmup"]),
        },
        "new_neffs": new_neffs,
        "cold_warmup_forward_s": cold_warmup_s,
        "snapshots": {
            name: {**{k: v for k, v in snap.items() if k != "cache_entries"},
                   "cache_entry_count": None if snap["cache_entries"] is None
                   else len(snap["cache_entries"])}
            for name, snap in snaps.items()
        },
        "sysfs_device_memory": sysfs_device_memory(logical_core),
        "outputs": {f"{ts:g}": {"sha256": sha256(out), "shape": list(out.shape)}
                    for ts, out in outputs.items()},
        "compiled_blocks": len(app.compiled_blocks),
        "load_report": {k: len(v) for k, v in report.items()},
    }
    (args.out_dir / f"result-rank{rank}.json").write_text(json.dumps(result, indent=2))

    if rank == 0:
        for name, passed, detail in checks:
            print(f"{'PASS' if passed else 'FAIL'} {name:26s} {detail}", flush=True)
    failed = [name for name, passed, _ in checks if not passed]
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} {args.exec_mode}"
          f"{'/' + args.expect if args.expect else ''} warmup_forward={warmup_s:.1f}s "
          f"forwards={[round(s, 2) for s in forward_s]} run_counters={run_delta} "
          f"new_neffs={new_neffs} max_memory_allocated="
          f"{result['max_memory_allocated_gib']:.2f}GiB"
          + (f" failed={failed}" if failed else ""), flush=True)
    if ok and dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
