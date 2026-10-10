"""Manual check: per-block torch.compile on the neuron device and the persistent NEFF cache.

Run on a trn2 host with the TorchNeuron stack, twice, against one cache directory: the first
run (cold) must find it empty and fill it, the second (warm) must start from it. From the
repository root, with any fresh directory as CACHE:

    CACHE=/var/tmp/difflet_c9/neff
    rm -rf "$CACHE" "${CACHE}_local" "$CACHE.cold.json"    # forces the next run to be cold
    DIFFLET_BACKEND=neuron DIFFLET_DISABLE_PREWARM=1 NEURON_RT_NUM_CORES=4 PYTHONPATH=. \
        torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_compile_c9.py \
        --cache-dir "$CACHE" --expect cold
    DIFFLET_BACKEND=neuron DIFFLET_DISABLE_PREWARM=1 NEURON_RT_NUM_CORES=4 PYTHONPATH=. \
        torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_compile_c9.py \
        --cache-dir "$CACHE" --expect warm

The NEFFs land in CACHE, the compile locks in CACHE_local, and the cold run's first-call time
in CACHE.cold.json (read by the warm run).

Model: six identical DiT-like blocks (LayerNorm, column-parallel q/k/v, attention through the
neuron op, row-parallel output, column/row MLP), bf16, TP = world size, sequence 512, eight
heads of 128. Rank 0 prints one PASS/FAIL line per case:

* explain: Dynamo traces the uncompiled model as one graph with no break, NKI attention call
  included;
* one_graph: compile_blocks hands the neuron backend one graph for all six blocks, and a second
  forward hands it none;
* nki_attention_in_graph / all_reduce_in_graph: that graph calls the NKI kernel op directly and,
  at TP > 1, the functional all-reduce of both row-parallel layers;
* parity: compiled matches eager on the device to bf16 tolerance; repeated compiled calls are
  bit-identical;
* fallbacks: no op falls back to CPU;
* fallback_execution_off: the neuron backend received options={"fallback_execution": False}
  (NEURON_COMPILE_OPTIONS), so a graph that fails to lower or run raises instead of silently
  running op by op;
* compiled_neff: during the first compiled forward this rank's runtime compiled a NEFF or
  loaded one from the persistent cache (CompilationCache.TotalCompilations + PersistentHits
  >= 1), so the forward ran a compiled graph; checked on every rank;
* neff_cache_cold: the compiled forward stores at least one NEFF in --cache-dir, and the lock
  directory exists where configure_compile_cache pointed it;
* neff_cache_warm / warm_start_time: the compiled forward stores no NEFF, compiles nothing and
  counts persistent-cache hits on every rank, and the first compiled forward takes under half
  the cold run's time;
* two_models_one_graph / two_models_options / two_models_parity: two more models of the same
  block class, each compiled by its own ``compile_blocks(model)`` call on the backend="neuron"
  string path (Wan 2.2 A14B's two transformers): only the first model's forward hands the
  backend a Dynamo graph, the second's hands it none and compiles or loads no NEFF; every block
  of both models got the one options dict the neuron backend edited; both match their eager
  forward with no CPU fallback. Checked on every rank.

Each rank then prints its own counters and ``[rank r] PASS`` (or FAIL).
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

from difflet.backends.neuron.compile import (
    NEURON_COMPILE_OPTIONS,
    compile_blocks,
    configure_compile_cache,
    explain_graphs,
)
from difflet.backends.neuron.ops_impl.collectives import init_parallel_mesh
from difflet.backends.neuron.runtime import track_fallbacks
from difflet.backends.registry import get_backend
from difflet.pipeline.parallel_config import DiffletParallelConfig
from tests.unit.backends._neuron_toy import ToyBlocksModel

N_BLOCKS, DIM, HIDDEN, HEADS, SEQ = 6, 1024, 2048, 8, 512
PARITY_TOL = 2e-2
NKI_ATTENTION_OP = "nki_kernels.scaled_dot_product_attention_kernel"
NEFF_COUNTERS = (
    "CompilationCache.TotalCompilations",
    "CompilationCache.PersistentHits",
    "CompilationCache.InMemoryHits",
)


class CountingBackend:
    """The neuron dynamo backend, keeping every graph Dynamo hands it and the options it got."""

    def __init__(self):
        self.inner = torch._dynamo.lookup_backend("neuron")
        self.graphs = []
        self.options = []

    def __call__(self, gm, example_inputs, **kwargs):
        self.graphs.append(gm)
        self.options.append(kwargs.get("options"))
        return self.inner(gm, example_inputs, **kwargs)


def count_neffs(directory: Path) -> int:
    return sum(1 for _ in directory.rglob("*.neff")) if directory.is_dir() else 0


def sync(world: int) -> None:
    """Wait for every rank's device work (an eager all-reduce on the default group)."""
    torch.neuron.synchronize()
    if world > 1:
        import torch.distributed as dist

        flag = torch.ones(1, device="neuron")
        dist.all_reduce(flag)
        torch.neuron.synchronize()


def rel_err(out, ref) -> float:
    out, ref = out.float().cpu(), ref.float().cpu()
    return ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()


def cache_counter(name: str) -> int:
    import torch_neuronx.metrics as metrics

    return metrics.get_counter_value(name) or 0


def block_options(block: torch.nn.Module):
    """The options dict Dynamo hands the backend for a block compiled in place (as C8's check
    reads it: compiled fn closure -> OptimizeContext -> ConvertFrameAssert -> the
    _TorchCompileWrapper), so the blocks keep the exact backend="neuron" string path."""
    try:
        ctx = next(
            cell.cell_contents
            for cell in block._compiled_call_impl.__closure__
            if type(cell.cell_contents).__name__ == "OptimizeContext"
        )
        wrapper = ctx.callback._torchdynamo_orig_backend._torchdynamo_orig_backend
        return wrapper.kwargs.get("options")
    except Exception as exc:  # noqa: BLE001 - reported as a FAIL by the caller
        return f"<unreadable: {exc!r}>"


def check_two_models(x) -> list[tuple[str, bool, str]]:
    """Two more models of the same block class, each compiled by its own
    ``compile_blocks(model)`` call (backend="neuron", default options), as two applications'
    loads would (Wan 2.2 A14B's two transformers). The first model's forward must hand the
    neuron backend one new Dynamo graph for its blocks; the second model's must hand it none
    and compile or load no NEFF, because both calls hand the backend one options dict, which
    the backend has edited. The runtime's in-memory NEFF lookups are reported too: one per
    block call, plus the new graph's own lookups in the first model only. Run after the main
    case, whose callable-backend graph stays in Dynamo's cache and does not match the
    "neuron" string."""
    from torch._dynamo.utils import counters

    models = []
    for seed in (2, 3):
        torch.manual_seed(seed)
        model = ToyBlocksModel(N_BLOCKS, DIM, HIDDEN, heads=HEADS, dtype=torch.bfloat16)
        models.append(model.eval().requires_grad_(False).to("neuron"))
    with torch.no_grad():
        refs = [model(x) for model in models]
    torch.neuron.synchronize()
    graphs, neffs, errs, fallbacks = [], [], [], []
    for model, ref in zip(models, refs):
        compile_blocks(model)
        graphs_before = counters["stats"]["unique_graphs"]
        before = [cache_counter(name) for name in NEFF_COUNTERS]
        with torch.no_grad(), track_fallbacks() as model_fallbacks:
            out = model(x)
        torch.neuron.synchronize()
        graphs.append(counters["stats"]["unique_graphs"] - graphs_before)
        neffs.append(tuple(cache_counter(n) - b for n, b in zip(NEFF_COUNTERS, before)))
        errs.append(rel_err(out, ref))
        fallbacks.extend(model_fallbacks)
    options = [block_options(block) for model in models for block in model.blocks]
    shared = options[0]
    same_dict = all(o is shared for o in options)
    edited = isinstance(shared, dict) and shared.get("fallback_execution") is False and (
        "dynamic" in shared  # written by the neuron backend: it received this very dict
    )
    return [
        (
            "two_models_one_graph",
            graphs == [1, 0] and neffs[1][:2] == (0, 0),
            f"graphs={graphs} neffs(compiled,persistent,in_memory)={neffs}",
        ),
        (
            "two_models_options",
            same_dict and edited,
            f"one_dict={same_dict} blocks={len(options)} options={shared}",
        ),
        (
            "two_models_parity",
            max(errs) <= PARITY_TOL and not fallbacks,
            f"rel_err={[f'{e:.2e}' for e in errs]} (tol {PARITY_TOL:.0e}) fallbacks={fallbacks}",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expect", choices=("cold", "warm"), required=True)
    args = parser.parse_args()
    cache_dir = args.cache_dir.expanduser().absolute()
    local_dir = cache_dir.with_name(cache_dir.name + "_local")
    timing_file = cache_dir.with_name(cache_dir.name + ".cold.json")

    # Before any device work: the runtime's compilation cache reads these once.
    configure_compile_cache(cache_dir, local_cache_dir=local_dir)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    init_parallel_mesh(parallel)

    import torch.distributed as dist
    import torch_neuronx.metrics as metrics

    rank = dist.get_rank() if dist.is_initialized() else 0
    metrics.set_enabled(True)
    results: list[tuple[str, bool, str]] = []

    torch.manual_seed(0)
    model = ToyBlocksModel(N_BLOCKS, DIM, HIDDEN, heads=HEADS, dtype=torch.bfloat16)
    model = model.eval().requires_grad_(False).to("neuron")
    x = torch.randn(1, SEQ, DIM, generator=torch.Generator().manual_seed(1))
    x = x.to(torch.bfloat16).to("neuron")

    with torch.no_grad(), track_fallbacks() as eager_fallbacks:
        ref = model(x)
    with torch.no_grad():
        graphs, breaks = explain_graphs(model, x)
    results.append(("explain", (graphs, breaks) == (1, 0), f"graphs={graphs} breaks={breaks}"))

    backend = CountingBackend()
    names = compile_blocks(model, backend=backend, options=NEURON_COMPILE_OPTIONS)
    sync(world)
    neffs_before = count_neffs(cache_dir)
    hits_before = cache_counter("CompilationCache.PersistentHits")
    compiles_before = cache_counter("CompilationCache.TotalCompilations")
    start = time.perf_counter()
    with torch.no_grad(), track_fallbacks() as compiled_fallbacks:
        out = model(x)
    first_call_s = time.perf_counter() - start
    sync(world)
    new_neffs = count_neffs(cache_dir) - neffs_before
    hits = cache_counter("CompilationCache.PersistentHits") - hits_before
    compiles = cache_counter("CompilationCache.TotalCompilations") - compiles_before
    with torch.no_grad():
        again = model(x)
    torch.neuron.synchronize()

    targets = []
    if backend.graphs:
        graph = backend.graphs[0].graph
        targets = [str(n.target) for n in graph.nodes if n.op == "call_function"]
    results.append((
        "one_graph",
        len(names) == N_BLOCKS and len(backend.graphs) == 1,
        f"blocks={len(names)} graphs={len(backend.graphs)} after two forwards",
    ))
    kernel_calls = sum(NKI_ATTENTION_OP in t for t in targets)
    results.append(("nki_attention_in_graph", kernel_calls >= 1, f"kernel calls={kernel_calls}"))
    if world > 1:
        reduces = sum("all_reduce" in t for t in targets)
        results.append(("all_reduce_in_graph", reduces >= 2, f"all_reduce nodes={reduces}"))
    err = rel_err(out, ref)
    repeat_ok = torch.equal(again.cpu(), out.cpu())
    results.append((
        "parity",
        err <= PARITY_TOL and repeat_ok,
        f"rel_err={err:.2e} (tol {PARITY_TOL:.0e}) repeat_identical={repeat_ok}",
    ))
    results.append((
        "fallbacks",
        not eager_fallbacks and not compiled_fallbacks,
        f"eager={eager_fallbacks} compiled={compiled_fallbacks}",
    ))
    results.append((
        "fallback_execution_off",
        bool(backend.options)
        and all(o is not None and o.get("fallback_execution") is False for o in backend.options),
        f"backend options={backend.options}",
    ))
    counters = f"compilations={compiles} persistent_hits={hits}"
    results.append(("compiled_neff", compiles + hits >= 1, counters))
    if args.expect == "warm":
        results.append(("warm_counters", compiles == 0 and hits >= 1, counters))
    results.extend(check_two_models(x))

    if rank == 0:
        detail = (
            f"new_neffs={new_neffs} persistent_hits={hits} compilations={compiles} "
            f"first_call={first_call_s:.1f}s lock_dir={local_dir.is_dir()} cache_dir={cache_dir}"
        )
        if args.expect == "cold":
            timing_file.write_text(json.dumps({"first_call_s": first_call_s}))
            results.append(("neff_cache_cold", new_neffs >= 1 and local_dir.is_dir(), detail))
        else:
            cold_s = None
            if timing_file.exists():
                cold_s = json.loads(timing_file.read_text())["first_call_s"]
            results.append(("neff_cache_warm", new_neffs == 0 and hits >= 1, detail))
            cold_txt = "missing" if cold_s is None else f"{cold_s:.1f}s"
            results.append((
                "warm_start_time",
                cold_s is not None and first_call_s < 0.5 * cold_s,
                f"warm={first_call_s:.1f}s cold={cold_txt}",
            ))

    ok = all(passed for _, passed, _ in results)
    if rank == 0:
        for case, passed, detail in results:
            print(f"{'PASS' if passed else 'FAIL'} {case:24s} {detail}", flush=True)
    failed = [case for case, passed, _ in results if not passed]
    print(
        f"[rank {rank}] {'PASS' if ok else 'FAIL'} {counters} first_call={first_call_s:.1f}s"
        + (f" failed={failed}" if failed else ""),
        flush=True,
    )
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
