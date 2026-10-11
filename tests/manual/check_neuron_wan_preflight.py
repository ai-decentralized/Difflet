"""Manual check (W1b, W2, W3): Wan2.1 T2V DiT through the neuron application on 4 ranks.

W3 (Gate 2) runs the 14B in compile mode: cold (empty NEFF cache) then warm, both with
``--eager-dir`` pointing at the W2 eager run.

Written as the pre-flight of the 14B gates: the identical code path
(``NeuronWanTransformerApplication``, TP4, the one static shape 480x832x9 = latent
(1,16,3,60,104) = 4,680 DiT tokens, 512 text tokens, the shared block graph in compile mode) on
the 1.3B checkpoint (30 blocks, dim 1536, 12 heads -> 3 per rank), so the cold block-graph
compile, per-rank HBM, the one-graph property and parity with the diffusers fp32 reference are
measured in minutes instead of hours. The same script runs the 14B gates (40 blocks, dim 5120,
40 heads -> 10 per rank; the model shape comes from the snapshot's config): Gate 1 (W2) is
``--exec-mode eager``, Gate 2 (W3) the compile cold/warm pair. ``--num-frames 5`` (480x832x5,
3,120 tokens) is the decision-D48 fallback shape when 480x832x9 exceeds 18 GiB per rank.

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

Each rank enables ``torch_neuronx.metrics`` and counts the ``.neff`` files of the cache before
``prepare_runtime`` (which itself compiles or loads one NEFF for the process group; otherwise
the compilation counters miss it, read 0 if never enabled, and "warm" proves nothing), loads with
``load(skip_warmup=True)`` and then runs ``warmup()`` (the two halves of ``load()``, split only
to read the counters in between), runs one DiT forward per timestep through the composed
application's host<->device adapter under ``track_fallbacks()``, and writes
``<out-dir>/result-rank<r>.json``: the checks, the parity metrics, ``parity.device_evidence``,
``max_memory_allocated`` at start (after ``prepare_runtime``, before the load), after the load,
the warm-up and the forwards, ``phase_seconds``, the compilation counters /
``CompilationCache.CompilationTime`` and ``.NeffSize`` / Dynamo graph counts / NEFF count /
device memory / host memory (VmHWM, VmRSS, MemAvailable, Cached) before ``prepare_runtime``, at
start, after load, after the warm-up and at the end, the ``.neff`` files the run added, the
runtime cache entries each phase added (per-NEFF size and compile time), in compile mode the
compile breakdown (Dynamo's timed phases and torch-neuronx's per-graph FX-pass / lowering /
NEFF-compile times, ``compile_breakdown``) and the block NEFF's entry, the driver's per-core
device memory counters, the process-group timeout, and the sha256 of its outputs. Rank 0 also
writes its outputs as fp32 on the host, ``<out-dir>/out-t<timestep>.pt``. CompilationTime sums
samples per rank; compiles overlap (within a rank and across ranks), so a sum is compiler work,
not wall clock: the wall-clock compile cost is the warm-up delta, cold minus warm (rank-max).
For the same breakdown as CSV, launch with ``TORCH_NEURONX_ENABLED_METRIC_TABLES=graph_stats``
and ``TORCH_NEURONX_METRICS_DIR=<dir>`` (torch-neuronx writes
``metric_table_graph_stats_rank<r>.csv`` at exit).

Checks (on every rank; rank 0 prints one PASS/FAIL line per check, each rank ``[rank r] PASS``):

* checkpoint: every parameter found, no unexpected tensor; heads_per_rank: heads / 4;
* fallbacks: ``track_fallbacks()`` around the warm-up and around the forwards is ``[]``;
  unwarmed_shapes: ``[]`` (recorded by the application in compile mode only);
  forward_shapes: every forward's device inputs have the warm-up's shapes (in eager mode too);
* parity_fp32_t<ts>: ``parity.check`` against the fp32 reference with the bf16 control, at the
  TPU thresholds (cosine >= 0.995, rel_l1 <= 1.15x the control's, same-dtype <= 1.2x);
* no_compile_in_forwards (every launch, every mode): the forwards neither compiled nor loaded a
  NEFF (``TotalCompilations + PersistentHits == 0``; a NEFF the warm-up never loaded shows as a
  persistent hit) and, in compile mode, added no Dynamo graph;
* instrumentation (W2 review): the evidence was live, not vacuous: the counters were enabled
  before ``prepare_runtime`` on a runtime not yet started, the forwards registered in-memory
  NEFF hits, and ``max_memory_allocated`` is above 0;
* hbm_d48: ``max_memory_allocated`` at the end at most 18 GiB (decision D48: above it, or on an
  OOM, the 14B gates continue at 480x832x5);
* compile mode: compiled_blocks (all of them), one_graph (Dynamo ``unique_graphs == 1`` after
  the warm-up and after the forwards, no graph break), warmup_neff (the warm-up compiled or
  loaded a NEFF), block_neff (each Dynamo graph's NEFF is a runtime cache entry the warm-up
  added; its size and this rank's compile time, 0 when loaded from disk, are printed); the
  cache entries each phase added are in the JSON;
* ``--eager-dir``: vs_eager_t<ts>, a sanity guard of the compiled output against the eager one:
  cosine >= 0.999 and rel_l1 <= 2.0x the bf16 control's rel_l1 (vs fp32) at that timestep;
  rel_max is reported, not gated. Compile mode's parity gate is parity_fp32_t<ts>, the same
  ``parity.check`` as eager (decision D58). Why not C9's ``rel_max <= 2e-2``: that bound came
  from a 6-block toy and lies below the bf16 noise floor of a 30-40 block DiT. Measured on the
  1.3B at t=996 / 500: the bf16 control is itself rel_max 2.5e-2 / 3.0e-2 from fp32; diffusers
  bf16 at 112 vs 12 CPU threads differs by rel_max 1.9e-2; a memory-layout change alone moved
  eager by rel_max 1.7e-2 / 1.9e-2; per block, compiled vs eager on the same input differs by
  rel_l1 2.5e-3 to 4.1e-3 (bf16 epsilon 3.9e-3), and the compiled stack ends closer to fp32
  than eager (0.77x / 0.74x vs 1.02x the control's rel_l1), at rel_max 3.0e-2 / 3.7e-2 from it;
* ``--expect cold``: rank 0 finds new ``.neff`` files and the lock directory; ``--expect warm``:
  no compilation, at least one persistent hit and no new ``.neff`` on every rank, and the
  warm-up under half the cold one; ``--cold-dir``: outputs ``torch.equal`` to the cold run's.
  ``--expect`` requires ``--eager-dir``, so a cold or warm compile launch is never judged
  without the vs-eager guard.

The reference is read and checked (its timesteps, its frame count against ``--num-frames``)
before any device work, so a wrong ``--ref`` fails in seconds rather than after the 14B load.

The process group is destroyed only when every check passed (a failing rank exits non-zero and
torchrun stops its peers). ``--inject-fail-rank R`` tests exactly that (review focus 4 of the
Wan plan): rank R raises right after the warm-up, printing the epoch time, while its peers go on
into the forwards and wait in the compiled block's collectives; torchrun has to stop them within
``TORCH_ELASTIC_SHUTDOWN_TIMEOUT`` (30 s) plus a margin, not the process-group timeout.
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

#: D58: compiled vs eager is a sanity guard above the bf16 noise floor (see the docstring).
VS_EAGER_MIN_COSINE = 0.999
VS_EAGER_MAX_RATIO_VS_CONTROL = 2.0
#: Decision D48: above this per-rank peak (or on an OOM) the 14B gates run at 480x832x5.
HBM_LIMIT_GIB = 18.0
COMPILATION_TIME = "CompilationCache.CompilationTime"
NEFF_SIZE = "CompilationCache.NeffSize"
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
    p.add_argument("--inject-fail-rank", type=int, default=None,
                   help="this rank raises right after the warm-up while the others go on into "
                        "the forwards (torchrun must stop them); the launch then fails")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--text-seq-len", type=int, default=512)
    args = p.parse_args()
    if args.expect is not None and args.exec_mode != "compile":
        p.error("--expect applies to --exec-mode compile")
    if args.expect is not None and args.cache_dir is None:
        p.error("--expect needs --cache-dir (the directory whose cold/warm state is checked)")
    if args.expect is not None and args.eager_dir is None:
        p.error("--expect needs --eager-dir (a compile launch is judged against eager too)")
    try:
        args.timesteps = [float(t) for t in args.timesteps.split(",")]
    except ValueError:
        p.error(f"--timesteps must be comma-separated numbers, got {args.timesteps!r}")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.inject_fail_rank is not None and not 0 <= args.inject_fail_rank < world:
        p.error(f"--inject-fail-rank {args.inject_fail_rank} is not a rank of WORLD_SIZE={world}")
    return args


# ------------------------------------------------------------------------- measurements


def neff_files() -> dict[str, int]:
    """The ``.neff`` files of the persistent NEFF cache on disk: relative path -> bytes."""
    raw = os.environ.get(NEFF_CACHE_ENV)
    directory = Path(raw) if raw else None
    if directory is None or not directory.is_dir():
        return {}
    files = {}
    for path in directory.rglob("*.neff"):
        try:
            files[str(path.relative_to(directory))] = path.stat().st_size
        except OSError:  # removed while listing
            continue
    return files


def count_neffs() -> int:
    return len(neff_files())


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


def neff_size() -> dict[str, Any]:
    """``CompilationCache.NeffSize`` in bytes: the sum and count over every sample (one per NEFF
    this process compiled or loaded) and the largest retained samples."""
    import torch_neuronx.metrics as metrics

    data = metrics.get_metric_data(NEFF_SIZE)
    if not data:
        return {"total_bytes": 0, "samples": 0, "largest_bytes": []}
    values = sorted((int(s["value"]) for s in data["samples"]), reverse=True)
    return {"total_bytes": int(data["accumulator"]), "samples": data["total_samples"],
            "retained": len(values), "largest_bytes": values[:8]}


def compile_breakdown() -> dict[str, Any]:
    """Where this process's compile time went (compile mode; errata P2 / W6's breakdown).

    ``dynamo_phases_s``: Dynamo's ``compilation_time_metrics``, seconds per timed phase summed
    over calls (Dynamo trace, AOT, the neuron backend's ``torch_neuronx_fx_passes`` and
    ``torch_neuronx_lower`` StableHLO lowering, and ``torch_neuronx_compile``, the NEFF compile
    times read back from the runtime's cache entries, 0 for a NEFF loaded from disk).
    ``neuron_graphs``: torch-neuronx's record per compiled FX graph (``get_dynamo_metrics``;
    ``graph_id`` is the NEFF cache key; FX-pass / lowering / compile microseconds), the same
    rows ``TORCH_NEURONX_ENABLED_METRIC_TABLES=graph_stats`` writes to CSV at exit. Compile
    times of concurrent compiles overlap: none of this is wall clock."""
    from dataclasses import asdict

    from torch._dynamo.utils import compilation_time_metrics
    from torch_neuronx.neuron_dynamo_backend import metrics as dynamo_metrics

    try:
        phases = {}
        for name, values in list(compilation_time_metrics.items()):
            values = [float(v) for v in values]
            phases[name] = {"total_s": sum(values), "count": len(values),
                            "max_s": max(values, default=0.0)}
        graphs = [asdict(m) for m in dynamo_metrics.get_dynamo_metrics()]
    except Exception as exc:  # noqa: BLE001 - recorded, and the block_neff check then fails
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"dynamo_phases_s": phases, "neuron_graphs": graphs}


def block_neffs(graphs: list[dict[str, Any]],
                added: dict[str, list[dict[str, Any]] | None]) -> list[dict[str, Any]]:
    """The runtime cache entries of the Dynamo-compiled graphs (the block NEFF), with the phase
    that added them: ``compilation_time_ms`` > 0 means this process compiled it, 0 that it was
    loaded from the persistent cache."""
    keys = {g["graph_id"] for g in graphs}
    return [{"phase": phase, **entry} for phase, entries in added.items()
            for entry in (entries or []) if entry["key"] in keys]


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
    """This process's VmHWM / VmRSS and the host's MemAvailable / Cached, in GiB: the 14B
    shards are 57 GB of fp32 that every rank slices while the page cache holds them."""
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmHWM:", "VmRSS:")):
            key, value = line.split(":", 1)
            values[f"{key}_gib"] = int(value.split()[0]) / 2**20  # kB
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:", "Cached:")):
            key, value = line.split(":", 1)
            values[f"host_{key}_gib"] = int(value.split()[0]) / 2**20  # kB
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


def snapshot(t0: float, *, device: bool = True, runtime_started: bool = True) -> dict[str, Any]:
    """Counters, timers and memory now. Before ``prepare_runtime`` pass
    ``runtime_started=False``: the runtime's in-memory compilation cache is then not read (its
    accessor builds the runtime's resource manager, which must not exist before the core is
    bound) and is empty in a fresh process anyway."""
    return {
        "t_s": time.perf_counter() - t0,
        "counters": compilation_counters(),
        "compilation_time": compilation_time(),
        "neff_size": neff_size(),
        "dynamo": dynamo_stats(),
        "neff_count": count_neffs(),
        "device_memory": device_memory() if device else None,
        "host_memory": host_memory(),
        "cache_entries": cache_entries() if runtime_started else {},
    }


def added_entries(after: dict[str, Any], before: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The cache entries a phase added (largest NEFF first)."""
    if after["cache_entries"] is None or before["cache_entries"] is None:
        return None
    added = [{"key": k, **v} for k, v in after["cache_entries"].items()
             if k not in before["cache_entries"]]
    return sorted(added, key=lambda e: -(e["neff_size_bytes"] or 0))


def instrumentation_check(metrics_setup: dict[str, bool], forwards_delta: dict[str, int],
                          peak_gib: float) -> tuple[bool, str]:
    """Whether the evidence the other checks read was live (W2 review): the counters were
    enabled before ``prepare_runtime`` on a runtime not yet started (else they read 0 and "no
    compilation" proves nothing), the forwards registered in-memory NEFF hits (they ran on the
    device through the counted cache), and the allocator reported a peak (the HBM numbers are
    not the 0 of an unstarted allocator)."""
    enabled = bool(metrics_setup.get("enabled_before_prepare_runtime"))
    fresh = metrics_setup.get("runtime_initialized_when_enabled") is False
    hits = int(forwards_delta.get("InMemoryHits", 0))
    ok = enabled and fresh and hits > 0 and peak_gib > 0
    return ok, (f"metrics enabled before prepare_runtime={enabled} on a fresh runtime={fresh}; "
                f"forward InMemoryHits={hits} (> 0); max_memory_allocated={peak_gib:.2f} GiB "
                f"(> 0)")


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


def _progress(rank: int, what: str, app: Any, snap: dict[str, Any]) -> None:
    """One line per rank per phase, so a long load can be followed in the log."""
    mem, host = snap["device_memory"], snap["host_memory"]
    print(f"[rank {rank}] {what} at {snap['t_s']:.1f}s: "
          f"phases={ {k: round(v, 1) for k, v in app.phase_seconds.items()} } "
          f"HBM allocated={mem['allocated_gib']:.2f} max={mem['max_allocated_gib']:.2f} GiB "
          f"VmHWM={host['VmHWM_gib']:.2f} GiB host MemAvailable="
          f"{host['host_MemAvailable_gib']:.1f} GiB Cached={host['host_Cached_gib']:.1f} GiB "
          f"counters={snap['counters']}", flush=True)


# ------------------------------------------------------------------------------- run


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    # The reference first (host only, seconds): a wrong --ref must not cost the 14B load.
    blob = torch.load(args.ref, weights_only=False)
    runs = {float(run["timestep"][0]): run for run in blob["runs"]}
    missing = [ts for ts in args.timesteps if ts not in runs]
    if missing:
        raise ValueError(f"{args.ref} has no reference at timesteps {missing} "
                         f"(it holds {sorted(runs)})")
    if blob.get("frames") != args.num_frames:
        raise ValueError(f"{args.ref} was computed at {blob.get('frames')} frames, this run "
                         f"is --num-frames {args.num_frames}; recompute the reference")
    for flag, directory in (("--eager-dir", args.eager_dir), ("--cold-dir", args.cold_dir)):
        absent = [f"out-t{ts:g}.pt" for ts in args.timesteps
                  if directory is not None and not (directory / f"out-t{ts:g}.pt").is_file()]
        if absent:
            raise FileNotFoundError(f"{flag} {directory} has no {absent}")
    cache_dir = local_dir = timing_file = None
    if args.cache_dir is not None:
        cache_dir = args.cache_dir.expanduser().absolute()
        local_dir = cache_dir.with_name(cache_dir.name + "_local")
        timing_file = cache_dir.with_name(cache_dir.name + ".cold.json")
        # Before any device work: the runtime's compilation cache reads these once.
        configure_compile_cache(cache_dir, local_cache_dir=local_dir)
    else:
        configure_compile_cache()  # what prepare_runtime does; resolved now to count the NEFFs

    # Before prepare_runtime, which itself compiles or loads one NEFF: enable the counters
    # (D53; they read 0 until enabled) and list the cache on disk. Neither starts the runtime;
    # prepare_runtime refuses to run if something did.
    import torch_neuronx
    import torch_neuronx.metrics as metrics

    metrics.set_enabled(True)
    metrics_setup = {"enabled_before_prepare_runtime": bool(metrics.is_enabled()),
                     "runtime_initialized_when_enabled":
                         bool(torch_neuronx.is_neuron_runtime_initialized())}
    neffs_before = neff_files()
    snaps = {"pre_prepare": snapshot(t0, device=False, runtime_started=False)}
    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    prepare_s = time.perf_counter() - t0

    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    logical_core = runtime.parse_core_list(os.environ.get(runtime.VISIBLE_CORES_ENV))
    logical_core = logical_core[0] if len(logical_core) == 1 else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Device memory too: the baseline after prepare_runtime, before the load (W2 review).
    snaps["start"] = snapshot(t0)

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
    _progress(rank, "loaded", app, snaps["after_load"])
    with track_fallbacks() as warmup_fallbacks:
        app.warmup()
    snaps["after_warmup"] = snapshot(t0)
    _progress(rank, "warmed up", app, snaps["after_warmup"])
    breakdown = {"after_warmup": compile_breakdown()} if args.exec_mode == "compile" else {}
    if rank == args.inject_fail_rank:
        # Review focus 4: a rank dies after the warm-up while its peers go on into the
        # forwards' collectives; torchrun must stop them (not the c10d timeout).
        print(f"[rank {rank}] injected failure at {time.time():.3f} (epoch s), "
              f"{time.strftime('%H:%M:%S')}", flush=True)
        raise RuntimeError(f"--inject-fail-rank {rank}: injected failure after the warm-up")

    forward_shapes: list[list[tuple[int, ...]]] = []

    def module(*device_args: torch.Tensor) -> torch.Tensor:
        # The device-side inputs of each forward, to compare with the warm-up's in every mode
        # (the application records unwarmed shapes in compile mode only).
        forward_shapes.append([tuple(a.shape) for a in device_args])
        return app(*device_args)

    adapter = _OnDeviceTransformer(module, config, torch.bfloat16, app.device)
    forward_s: list[float] = []
    adapter.forward_hook = forward_s.append
    outputs: dict[float, torch.Tensor] = {}
    with track_fallbacks() as fallbacks:
        for ts in args.timesteps:
            out = adapter(blob["hidden_states"], runs[ts]["timestep"],
                          blob["encoder_hidden_states"])
            outputs[ts] = out.float()
    snaps["end"] = snapshot(t0)
    if args.exec_mode == "compile":
        breakdown["end"] = compile_breakdown()
    evidence = parity.device_evidence(app, fallbacks)
    neffs_after = neff_files()

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
    checks.append(("forward_shapes",
                   len(forward_shapes) == len(args.timesteps)
                   and all(shapes == app.warmup_shapes for shapes in forward_shapes),
                   f"{len(forward_shapes)} forwards at {sorted(set(map(str, forward_shapes)))} "
                   f"warmup={app.warmup_shapes}"))

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

    prepare_delta = delta(snaps["start"], snaps["pre_prepare"])
    load_delta = delta(snaps["after_load"], snaps["start"])
    warmup_delta = delta(snaps["after_warmup"], snaps["after_load"])
    forwards_delta = delta(snaps["end"], snaps["after_warmup"])
    run_delta = delta(snaps["end"], snaps["pre_prepare"])
    added_neffs = sorted(set(neffs_after) - set(neffs_before))
    new_neffs = len(added_neffs)
    added = {
        "prepare_runtime": added_entries(snaps["start"], snaps["pre_prepare"]),
        "load": added_entries(snaps["after_load"], snaps["start"]),
        "warmup": added_entries(snaps["after_warmup"], snaps["after_load"]),
        "forwards": added_entries(snaps["end"], snaps["after_warmup"]),
    }
    graphs_warm = snaps["after_warmup"]["dynamo"]
    graphs_end = snaps["end"]["dynamo"]
    # Every launch, every mode (W1b review): the forwards run what the warm-up compiled or
    # loaded, nothing else. A NEFF the warm-up never loaded shows as a persistent hit here.
    forward_neffs = forwards_delta["TotalCompilations"] + forwards_delta["PersistentHits"]
    no_new_graph = (args.exec_mode != "compile"
                    or graphs_end["unique_graphs"] == graphs_warm["unique_graphs"])
    checks.append(("no_compile_in_forwards", forward_neffs == 0 and no_new_graph,
                   f"forward counters {forwards_delta}, unique_graphs "
                   f"{graphs_warm['unique_graphs']} -> {graphs_end['unique_graphs']}"))
    peak_gib = snaps["end"]["device_memory"]["max_allocated_gib"]
    checks.append(("instrumentation",
                   *instrumentation_check(metrics_setup, forwards_delta, peak_gib)))
    checks.append(("hbm_d48", peak_gib <= HBM_LIMIT_GIB,
                   f"max_memory_allocated start="
                   f"{snaps['start']['device_memory']['max_allocated_gib']:.2f} after load="
                   f"{snaps['after_load']['device_memory']['max_allocated_gib']:.2f} "
                   f"warm-up={snaps['after_warmup']['device_memory']['max_allocated_gib']:.2f} "
                   f"end={peak_gib:.2f} GiB (limit {HBM_LIMIT_GIB} GiB, decision D48)"))
    blocks: list[dict[str, Any]] = []
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
        # The block NEFF is the runtime cache entry of the Dynamo graph: one per graph, added
        # by the warm-up (compiled there, or loaded from disk on a warm launch), none later.
        graphs = breakdown["end"].get("neuron_graphs", [])
        blocks = block_neffs(graphs, added)
        checks.append(("block_neff",
                       len(graphs) == graphs_end["unique_graphs"] and len(blocks) == len(graphs)
                       and all(b["phase"] == "warmup" for b in blocks),
                       f"{len(graphs)} neuron graph(s), entries "
                       + str([(b["phase"], b["key"][:12], b["neff_size_bytes"],
                               round(b["compilation_time_ms"] / 1e3, 1)) for b in blocks])
                       + f" (phase, key, bytes, compile s){breakdown['end'].get('error', '')}"))
    else:
        checks.append(("eager_compiles_no_block", not app.compiled_blocks
                       and graphs_end["unique_graphs"] == 0,
                       f"compiled_blocks={len(app.compiled_blocks)} "
                       f"unique_graphs={graphs_end['unique_graphs']}"))

    if args.eager_dir is not None:
        # D58: a sanity guard above the bf16 noise floor, not a parity gate (see the docstring).
        for ts, out in outputs.items():
            eager = torch.load(args.eager_dir / f"out-t{ts:g}.pt", weights_only=True)
            m = parity.metrics(out, eager)
            control = parity_results[f"{ts:g}"]["control"]
            limit = VS_EAGER_MAX_RATIO_VS_CONTROL * control["rel_l1"]
            ratio = m["rel_l1"] / control["rel_l1"]
            parity_results[f"{ts:g}"]["vs_eager"] = m
            parity_results[f"{ts:g}"]["vs_eager_ratio_vs_control"] = ratio
            checks.append((f"vs_eager_t{ts:g}",
                           m["cosine"] >= VS_EAGER_MIN_COSINE and m["rel_l1"] <= limit,
                           f"cos={m['cosine']:.6f} (min {VS_EAGER_MIN_COSINE}) "
                           f"rel_l1={m['rel_l1']:.3e} ({ratio:.3f}x control "
                           f"{control['rel_l1']:.3e}, max {VS_EAGER_MAX_RATIO_VS_CONTROL}x) "
                           f"rel_max={m['rel_max']:.3e} (reported, not gated) "
                           f"bit_identical={torch.equal(out, eager)}"))

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
        "metrics_setup": metrics_setup,
        "pg_timeout": pg_timeout(),
        "env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "HF_HUB_OFFLINE", "DIFFLET_NO_JEMALLOC", runtime.PG_TIMEOUT_ENV,
            runtime.VISIBLE_CORES_ENV, "TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT",
            "DIFFLET_DISABLE_PREWARM", "TORCH_NEURONX_ENABLED_METRIC_TABLES",
            "TORCH_NEURONX_METRICS_DIR")},
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
        "max_memory_allocated_gib_by_phase": {
            name: snaps[name]["device_memory"]["max_allocated_gib"]
            for name in ("after_load", "after_warmup", "end")},
        "weights_allocated_gib": snaps["after_load"]["device_memory"]["allocated_gib"],
        "host_vmhwm_gib_by_phase": {name: snap["host_memory"]["VmHWM_gib"]
                                    for name, snap in snaps.items()},
        "phase_seconds": dict(app.phase_seconds),
        "timers_s": {"prepare_runtime": prepare_s, "forwards": forward_s,
                     "total": time.perf_counter() - t0},
        "counters": {"prepare_runtime": prepare_delta, "load": load_delta,
                     "warmup": warmup_delta, "forwards": forwards_delta, "run": run_delta},
        "cache_entries_added": added,
        "compile_breakdown": breakdown,
        "block_neffs": blocks,
        "device_memory_at_start": snaps["start"]["device_memory"],
        "forward_shapes": forward_shapes,
        "neffs_before_prepare_runtime": len(neffs_before),
        "new_neffs": new_neffs,
        "new_neff_files": {path: neffs_after[path] for path in added_neffs},
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
    block_line = ""
    if args.exec_mode == "compile":
        block_line = ("block_neff(bytes, compile s)="
                      + str([(b["neff_size_bytes"], round(b["compilation_time_ms"] / 1e3, 1))
                             for b in blocks]) + " ")
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} {args.exec_mode}"
          f"{'/' + args.expect if args.expect else ''} warmup_forward={warmup_s:.1f}s "
          f"load_checkpoint={app.phase_seconds['load checkpoint']:.1f}s "
          f"forwards={[round(s, 2) for s in forward_s]} run_counters={run_delta} "
          f"forward_counters={forwards_delta} new_neffs={new_neffs} {block_line}"
          f"max_memory_allocated start="
          f"{snaps['start']['device_memory']['max_allocated_gib']:.2f}GiB after_load="
          f"{snaps['after_load']['device_memory']['max_allocated_gib']:.2f}GiB end="
          f"{result['max_memory_allocated_gib']:.2f}GiB VmHWM="
          f"{snaps['end']['host_memory']['VmHWM_gib']:.2f}GiB"
          + (f" failed={failed}" if failed else ""), flush=True)
    if ok and dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
