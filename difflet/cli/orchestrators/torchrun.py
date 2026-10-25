"""Driver and rank helpers shared by the staged orchestrators that run on a non-AoT torchrun
backend (neuron): Wan (``wan.py``), Qwen-Image (``qwen_image.py``), Flux (``flux.py``) and
LTX-2 (``ltx_2.py``).

Every helper is inert unless DIFFLET_BACKEND names such a backend (``torchrun_backend``), so
unset, trainium and tpu keep their stage paths unchanged. Torch-free except the helpers that run
in the ranks (``fallback_tracker``, ``replica_requests``/``replica_request_scope``, a DP
replica's request queue, and the two stage bodies ``run_generate_stage`` and
``run_vae_stage``), which import torch when called: the driver checks must not import torch
before torchrun fans out.

On the driver side: the options every torchrun backend refuses (``torchrun_unsupported_options``)
and the refusals themselves (``refuse``, ``refuse_torchrun_conflicts``, with the conflicts two
models share: ``vae_placement_conflict``, ``cfg_cp_conflict``), the cores of one replica
(``replica_cores``), where the decode runs (``decodes_on_device``), the HBM checks before
torchrun starts (``refuse_dit_overflow``, ``device_encoder_overflow``) and the backend fields of
the stage cache keys (``torchrun_stage_keys``). What each model runs, refuses and writes stays in
its orchestrator.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from difflet import envs
from difflet.cli import runner


def torchrun_backend() -> str | None:
    """The non-AoT torchrun-MPMD backend DIFFLET_BACKEND names (neuron), else None.

    The launcher's own opt-in rule (``runner._torchrun_mpmd_selected``): exactly when a stage
    runs as torchrun ranks, the DiT stage builds the backend's application through
    DiffletPipeline and the stage cache key names the backend.
    """
    if not runner._torchrun_mpmd_selected():
        return None
    from difflet.backends import get_backend

    return get_backend(envs.DIFFLET_BACKEND).name


#: The TeaCache options as the command line names them, with their argparse destinations.
TEACACHE_FLAGS = (
    ("--teacache-cadence", "teacache_cadence"),
    ("--teacache-online-delta", "teacache_online_delta"),
    ("--teacache-speedup", "teacache_speedup"),
    ("--teacache-calibration", "teacache_calibration"),
)


def torchrun_unsupported_options(args: argparse.Namespace) -> list[str]:
    """The options every torchrun backend's ranks reject in phase 1, as the command line names
    them: TeaCache, and a --shapes with more than one shape (the applications hold one static
    shape per process, ``_resolve_shape``; a bucketed set used to fail only after torchrun had
    started every rank).

    The ranks keep their own checks (the application's ``_reject_unsupported``); this list lets
    the driver refuse the same options before torchrun starts tp*cp processes that import torch
    only to fail. A model adds its own (LTX-2: --cp-mode ring, --sp). Torch-free: the shapes are
    counted, not parsed, so a malformed --shapes fails where it always did.
    """
    found = [flag for flag, name in TEACACHE_FLAGS if getattr(args, name, None) is not None]
    shapes = getattr(args, "shapes", None)
    if shapes and len([token for token in str(shapes).split(",") if token.strip()]) > 1:
        found.append("--shapes with more than one shape")
    return found


def reject_unsupported_torchrun_options(found: list[str], *, runs: str) -> None:
    """On a torchrun backend, ``found`` non-empty: print one message naming the options and exit
    2 before anything is launched (or downloaded, for ``run``). ``runs`` is what the model's
    ranks do support, request batching and TeaCache included ("tensor and context parallelism
    at one static shape, one request per launch, no TeaCache"; Wan's, Qwen-Image's and Flux's
    DP workers drain a request queue, and Wan and Flux run TeaCache, so the phrase is the
    model's, not a fixed suffix). No-op on unset,
    trainium and tpu, which support these options or reject them where they always did."""
    if not found:
        return
    backend = torchrun_backend()
    if backend is None:
        return
    print(f"Error: the {backend} backend does not support {', '.join(found)} (it runs "
          f"{runs}); nothing was launched.",
          file=sys.stderr, flush=True)
    raise SystemExit(2)


def refuse(problem: str | None) -> None:
    """Print ``problem`` as the driver's one-line refusal and exit 2 before anything is launched
    (or downloaded, for ``run``); nothing when it is None."""
    if problem is None:
        return
    print(f"Error: {problem}; nothing was launched.", file=sys.stderr, flush=True)
    raise SystemExit(2)


def refuse_torchrun_conflicts(conflicts: list[str]) -> None:
    """On a torchrun backend, refuse the first of ``conflicts`` (``refuse``); the AoT backends
    keep their own paths for these flags."""
    if conflicts and torchrun_backend() is not None:
        refuse(conflicts[0])


def vae_placement_conflict(args: argparse.Namespace) -> str | None:
    """--host-vae with --device-vae. Torch-free."""
    if getattr(args, "host_vae", False) and getattr(args, "device_vae", False):
        return "--host-vae and --device-vae are mutually exclusive"
    return None


def cfg_cp_conflict(args: argparse.Namespace) -> str | None:
    """--cfg-parallel with --cp-degree > 1: both take the data-parallel lanes (the neuron
    configs refuse the pair in every rank as well). Torch-free."""
    if getattr(args, "cfg_parallel", False) and (getattr(args, "cp_degree", None) or 1) > 1:
        return "--cfg-parallel and --cp-degree > 1 are mutually exclusive"
    return None


def replica_cores(args: argparse.Namespace) -> int:
    """Cores of one DiT launch: tp x cp, doubled by --cfg-parallel (each CFG branch on its own
    TP group). Torch-free."""
    return ((args.tp_degree or 4) * (args.cp_degree or 1)
            * (2 if getattr(args, "cfg_parallel", False) else 1))


def decodes_on_device(args: argparse.Namespace, default: bool) -> bool:
    """Whether the neuron backend decodes in a vae stage on one core instead of on the host in
    the driver: --host-vae wins, then --device-vae, then the model's ``default``. Torch-free."""
    if getattr(args, "host_vae", False):
        return False
    if getattr(args, "device_vae", False):
        return True
    return default


def _local_snapshot(resolve_model_dir: Callable[[], str]) -> str | None:
    """The local snapshot, or None when the weights are not local yet (``compile`` before any
    download): the application's own check refuses later instead."""
    try:
        return resolve_model_dir()
    except (OSError, RuntimeError):
        return None


def refuse_dit_overflow(model: str, resolve_model_dir: Callable[[], str],
                        dit_fits: Callable[..., tuple[bool, int, int]], tp: int) -> None:
    """Refuse a TP degree whose DiT shard clearly cannot fit one core (``dit_fits``, the model's
    ``hbm.py``) before any rank starts: print one message and exit 2.

    The application base refuses it too, but only once torchrun has started every rank and each
    has built the model on meta; a DP router would start every replica for it. The per-rank
    share depends on the TP degree alone, so a DP replica of tp1 is caught the same way as a
    plain launch. Weights that are not local yet leave the decision to the application's check.
    Torch-free, like every driver check."""
    model_dir = _local_snapshot(resolve_model_dir)
    if model_dir is None:
        return
    fits, estimate, limit = dit_fits(model_dir, tp_degree=tp)
    if fits:
        return
    from difflet.backends.neuron.hbm import GIB, suggested_tp

    print(f"Error: the {model} DiT needs at least {estimate / GIB:.1f} GiB of bf16 weights per "
          f"core at tp={tp}, over the {limit / GIB:.1f} GiB budget of one NeuronCore (every DP "
          f"replica is its own TP group); use --tp-degree {suggested_tp(estimate, tp, limit)} "
          "or more; nothing was launched.", file=sys.stderr, flush=True)
    raise SystemExit(2)


def device_encoder_overflow(resolve_model_dir: Callable[[], str],
                            fits: Callable[..., tuple[bool, int, int]], tp: int) -> str | None:
    """Why a device text encoder cannot share the cores with the DiT at ``tp`` (``fits`` on the
    local snapshot, the model's ``hbm.py``), as "needs at least X GiB per core beside the DiT
    at tp=N, over the Y GiB budget" for the model's own line; None when it fits or the weights
    are not local yet (the application's own check still refuses). Torch-free."""
    model_dir = _local_snapshot(resolve_model_dir)
    if model_dir is None:
        return None
    ok, estimate, limit = fits(model_dir, tp_degree=tp)
    if ok:
        return None
    from difflet.backends.neuron.hbm import GIB

    return (f"needs at least {estimate / GIB:.1f} GiB per core beside the DiT at tp={tp}, "
            f"over the {limit / GIB:.1f} GiB budget")


def torchrun_stage_keys(inputs: dict) -> dict:
    """``inputs`` plus the backend and the exec mode on a torchrun backend, unchanged on the AoT
    backends. Additive: a torchrun stage dir never matches a Trainium/TPU artifact, and eager
    and compile differ (an omitted --exec-mode keys as the default, ``resolve_exec_mode``)."""
    backend = torchrun_backend()
    if backend is None:
        return inputs
    from difflet.backends.neuron.compile import resolve_exec_mode

    return {**inputs, "backend": backend, "exec_mode": resolve_exec_mode(envs.DIFFLET_EXEC_MODE)}


def root_compile_cache_at_cache_dir(cache_dir: str | None, *, tag: str) -> None:
    """A torchrun backend (neuron) compiles into the NEFF and HLO caches that
    ``runner.run_stage`` exports to its ranks under ``$DIFFLET_COMPILE_CACHE/neuron/``, while
    --cache-dir holds only the stage manifest. With --cache-dir given and DIFFLET_COMPILE_CACHE
    unset or empty, export --cache-dir (absolute) as DIFFLET_COMPILE_CACHE before any stage
    launch, so the compile cache lives under it; an exported DIFFLET_COMPILE_CACHE wins. Read
    from os.environ directly: envs fills in the default when the variable is unset. ``tag``
    prefixes the note (``[wan]``)."""
    if not cache_dir or os.environ.get("DIFFLET_COMPILE_CACHE"):
        return
    backend = torchrun_backend()
    if backend is None:
        return
    root = os.path.abspath(os.path.expanduser(cache_dir))
    os.environ["DIFFLET_COMPILE_CACHE"] = root
    print(f"[{tag}] backend {backend!r} compiles into --cache-dir: "
          f"DIFFLET_COMPILE_CACHE={root} (NEFF and HLO caches under {root}/neuron)",
          flush=True)


def fallback_tracker(pipe):
    """``track_fallbacks()`` for a pipeline on the neuron backend (it synchronizes the device);
    an empty record otherwise."""
    if getattr(getattr(pipe, "backend", None), "name", None) == "neuron":
        from difflet.backends.neuron.runtime import track_fallbacks

        return track_fallbacks()
    return contextlib.nullcontext([])


def replica_requests(args: argparse.Namespace, *, device):
    """The requests a torchrun replica runs, one at a time, every rank the same request.

    Outside batch mode, or with one rank, this is ``stage_loop.claim_requests``. In a DP worker
    (``--requests-dir``) under torchrun, rank 0 alone claims from the queue and broadcasts the
    claimed index (-1 when the queue is empty, as int32) to the replica's ranks, which read the
    request from the manifest: claiming on every rank would hand the TP group different requests.
    """
    from difflet.backends.neuron.core import distributed
    from difflet.cli.dp import stage_loop

    rank, world = distributed.world_info()
    if not stage_loop.batch_mode(args) or world == 1:
        yield from stage_loop.claim_requests(args)
        return
    import torch

    from difflet.cli.dp.requests_io import read_manifest

    manifest = {r.index: r for r in read_manifest(Path(args.requests_dir))}
    claims = stage_loop.claim_requests(args) if rank == 0 else None
    while True:
        req = next(claims, None) if rank == 0 else None
        index = (torch.tensor([req.index if req is not None else -1], dtype=torch.int32)
                 if rank == 0 else None)
        index = int(distributed.broadcast_tensor(index, device=device).reshape(-1)[0].item())
        if index < 0:
            return
        yield req if rank == 0 else manifest[index]


@contextlib.contextmanager
def replica_request_scope(args: argparse.Namespace, req, *, final: bool = False):
    """``stage_loop.request_scope`` for a torchrun replica: a failed request is marked failed
    once (rank 0) and every rank re-raises. A rank cannot skip a request its TP partners are
    still running: they would wait in a collective forever. With ``final`` (the stage that
    finishes a request), rank 0 marks a successful request done. One rank stays as before."""
    from difflet.backends.neuron.core import distributed
    from difflet.cli.dp import claims, stage_loop

    rank, world = distributed.world_info()
    if not stage_loop.batch_mode(args) or world == 1:
        with stage_loop.request_scope(args, req, final=final):
            yield
        return
    try:
        yield
    except Exception:
        if rank == 0:
            import traceback

            claims.mark_failed(args.requests_dir, req.index, traceback.format_exc())
            print(f"[dp-worker] request {req.index} failed; stopping this replica", flush=True)
        raise
    if final and rank == 0:
        claims.mark_done(args.requests_dir, req.index)


def torchrun_parallel(args: argparse.Namespace):
    """The whole parallel request of a stage (tp, cp and its mode, CFG, SP): the backend and the
    application refuse what they do not run, rather than the stage dropping it."""
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    return DiffletParallelConfig(
        tp_degree=args.tp_degree or 4,
        cp_degree=args.cp_degree or 1,
        cp_mode=getattr(args, "cp_mode", None) or "gather_kv",
        cfg_parallel_enabled=bool(getattr(args, "cfg_parallel", False)),
        sp_enabled=bool(getattr(args, "sp_enabled", False)),
    )


def run_generate_stage(args: argparse.Namespace, backend: str, *, tag: str, model_id: str,
                       pipeline_kwargs: dict[str, Any], application_kwargs: dict[str, Any],
                       request_kwargs: Callable[[Any], dict[str, Any]],
                       save: Callable[[Any, Any], None],
                       finish_compile: Callable[[], None], timing: bool = False,
                       final: bool = False) -> None:
    """The DiT stage on a non-AoT torchrun backend (neuron); runs on every rank.

    ``DiffletPipeline.from_pretrained(model_id, **pipeline_kwargs, ...)`` prepares the runtime
    (NEFF cache, core binding, the process group) before it builds the registry's application,
    and ``load(skip_warmup=False)`` compiles or loads the blocks at the static shape (and loads
    the text encoder), so the step loop compiles nothing. Nothing is built ahead of time:
    ``compile`` mode is that load plus the stage manifest (``finish_compile``, on rank 0);
    ``generate`` requires no artifact. Every rank runs the same request,
    ``pipe(**request_kwargs(req))``, and rank 0 alone writes its outputs (``save``): the work
    files and the compiled dir are the same paths on every rank. In a DP worker
    (``--requests-dir``) the launch is one replica draining the request queue: rank 0 alone
    claims and broadcasts each request's index (``replica_requests``), and a request that fails
    is marked failed once and stops every rank of the replica (``replica_request_scope``); with
    ``final`` (a stage that writes the request's output itself, such as a decode in process)
    rank 0 marks a successful one done.
    Every rank then prints one line of evidence: ``[<tag>] <backend> rank r/world:`` the exec
    mode, the compiled blocks, the unwarmed shapes, the CPU fallbacks, the load and stage
    seconds. The process group is destroyed after a successful stage only: a failing rank exits
    with its exception and torchrun stops its peers, which may be blocked in a collective.

    With ``timing`` (Wan), rank 0 also prints the start-up split (``from_pretrained`` against
    ``load``, with the application's ``phase_seconds`` by component) and, per request, the
    wall time, the time to the first step's end and the median step, read from the scheduler's
    ``step`` calls (one per denoising step, after that step's DiT forwards).
    """
    import time

    import torch
    import torch.distributed as dist

    from difflet.backends.neuron.compile import resolve_exec_mode
    from difflet.backends.neuron.core import distributed
    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    parallel = torchrun_parallel(args)
    start = time.perf_counter()
    pipe = DiffletPipeline.from_pretrained(
        model_id,
        local_files_only=True,
        parallel=parallel,
        dtype=torch.bfloat16,
        compile_cache_dir=args.cache_dir,
        backend=backend,
        application_kwargs=application_kwargs,
        load=False,
        **pipeline_kwargs,
    )
    built_s = time.perf_counter() - start  # prepare_runtime (the process group) and the build
    with fallback_tracker(pipe) as load_fallbacks:
        pipe.load(skip_warmup=False)
    load_s = time.perf_counter() - start
    if timing and distributed.is_rank0():
        components = getattr(pipe.app, "phase_seconds", {}) or {}
        print(f"[{tag}] start-up: from_pretrained (prepare_runtime + build) {built_s:.2f} s; "
              f"load {load_s - built_s:.1f} s by component {components}", flush=True)

    generate_fallbacks: list = []
    if args.stage_mode == "compile":
        if distributed.is_rank0():
            finish_compile()
    else:
        # Step timing without touching the shared orchestrator: the scheduler steps once per
        # denoising step, after that step's DiT forward(s).
        marks: list[float] = []
        scheduler = getattr(getattr(pipe.app, "pipeline", None), "scheduler", None)
        if timing and scheduler is not None:
            original_step = scheduler.step

            def timed_step(*a, **k):
                result = original_step(*a, **k)
                marks.append(time.perf_counter())
                return result

            scheduler.step = timed_step
        for req in replica_requests(args, device=getattr(pipe.app, "device", None)):
            with replica_request_scope(args, req, final=final):
                marks.clear()
                called = time.perf_counter()
                with fallback_tracker(pipe) as fallbacks:
                    out = pipe(**request_kwargs(req))
                generate_fallbacks.extend(fallbacks)
                done = time.perf_counter()
                if distributed.is_rank0() and len(marks) > 1:
                    import statistics

                    deltas = sorted(b - a for a, b in zip(marks, marks[1:]))
                    print(f"[{tag}] request: {done - called:.2f} s; to the first step's end "
                          f"{marks[0] - called:.2f} s (prompt encode + step 1); step median "
                          f"{statistics.median(deltas) * 1e3:.1f} ms (n={len(deltas)}, max "
                          f"{deltas[-1] * 1e3:.1f}); after the last step {done - marks[-1]:.2f} s",
                          flush=True)
                if distributed.is_rank0():
                    save(req, out)

    app = pipe.app
    rank, world = distributed.world_info()
    print(
        f"[{tag}] {backend} rank {rank}/{world}: "
        f"exec_mode={resolve_exec_mode(envs.DIFFLET_EXEC_MODE)} "
        f"compiled_blocks={len(getattr(app, 'compiled_blocks', []))} "
        f"unwarmed_shapes={list(getattr(app, 'unwarmed_shapes', []))} "
        f"fallbacks: load={list(load_fallbacks)} generate={generate_fallbacks} "
        f"load+warmup {load_s:.1f}s, stage {time.perf_counter() - start:.1f}s",
        flush=True,
    )
    if dist.is_initialized():
        dist.destroy_process_group()


def run_vae_stage(args: argparse.Namespace, *, build: Callable[[Any], Any],
                  finish_compile: Callable[[], None],
                  prepare_decode: Callable[[Any], Callable[[Any], None]]) -> None:
    """The neuron backend's vae stage: one rank, a launch of its own after the DiT stage has
    exited, so the decoder has a whole core whatever the DiT's topology was.

    ``build(parallel)`` returns the decoder at tp1 (an application, or the adapter around one;
    anything with ``load(skip_warmup=...)``). ``compile`` mode loads it with the warm-up decode,
    which compiles and caches the per-op kernels at the request shape, and writes the manifest
    (``finish_compile``, on rank 0); ``generate`` skips the warm-up, a second full decode that
    eager mode does not need (the request's decode compiles whatever the cache lacks), then
    decodes each request this replica finished with ``prepare_decode(decoder)(req)``, marking
    it done after its output is written (a failing one is marked failed, and the next one
    decodes).

    The neuron runtime is prepared first (``prepare_runtime``: the NEFF, lock and HLO cache
    locations under DIFFLET_COMPILE_CACHE, the core binding, the process group), as
    ``DiffletPipeline.from_pretrained`` prepares it for the DiT stage. The decoder applications
    are built directly, so without it a vae stage started outside ``runner.run_stage`` (which
    exports the cache locations to its ranks) would compile into torch-neuronx's default
    ``/tmp/neff_cache``.
    """
    from difflet.backends import get_backend
    from difflet.backends.neuron.core import distributed
    from difflet.cli.dp import stage_loop
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    parallel = DiffletParallelConfig(tp_degree=1)
    get_backend("neuron").prepare_runtime(parallel)
    decoder = build(parallel)
    decoder.load(skip_warmup=args.stage_mode != "compile")
    if args.stage_mode == "compile":
        if distributed.is_rank0():
            finish_compile()
        return
    decode = prepare_decode(decoder)
    for req in stage_loop.claimed_requests(args):
        with stage_loop.request_scope(args, req, final=True):
            decode(req)
