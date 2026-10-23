"""The one module on the TPU path that talks to a TPU runtime library.

Phase 1 of docs/plans/2026-10-22-torchtpu-migration.md. `torch_xla` is
deprecated and its repo is archived 2026-11-15, and TorchTPU's compilation
cache fixes the warm-start gap that `core/application_base.py` documents as
unfixable under torch_xla. Rather than edit the backend in place (which would
destroy the ability to A/B against the recorded `benchmark/v5e/` numbers) or
fork a second backend (which would mean forking the 1763 lines of
`models/*/tpu_application.py` where most of the call sites live), every TPU
call site routes through here and the runtime is chosen by one env var.

    DIFFLET_TPU_RUNTIME=torch_xla   (default, what ships today)
    DIFFLET_TPU_RUNTIME=torch_tpu   (Phase 2)

**Phase 1 changes no behavior.** Only the torch_xla implementations exist; the
torch_tpu ones raise, pointing at the phase that fills them in. The gate for
this phase is `benchmark.bench --backend tpu` reproducing the recorded numbers,
which it can only do if these wrappers are exactly what the call sites did
before — so each one documents the call it replaces.

Imports stay inside the functions, as they were at every call site before: the
TPU backend has to be importable on a host with no TPU runtime at all (the
CPU backend, unit tests on a laptop, `difflet plan`).

**Two torch_xla call sites are deliberately NOT routed through here**, both in
`difflet/models/flux/pipeline.py` (the `XLA_AVAILABLE` / `xm.mark_step()` pair
diffusers ships and this fork carried over). That file defines
`NeuronFluxPipeline`, which `difflet/models/flux/application.py` imports on
**Trainium**, and `requirements-neuron.lock` pins `torch-xla==2.9.0`, so
`is_torch_xla_available()` is true there and that `mark_step` is Trainium's,
not the TPU backend's. Routing it here would drag the production backend
through a TPU module for no gain. Leave it.
"""

from __future__ import annotations

import os
import pathlib

#: Selected once per process. Changing the env var mid-run does nothing, which
#: is deliberate: the runtime is process-global state and a mid-flight switch
#: would leave already-created tensors on the other one's device.
_RUNTIME: str | None = None

TORCH_XLA = "torch_xla"
TORCH_TPU = "torch_tpu"
_RUNTIMES = (TORCH_XLA, TORCH_TPU)


def runtime_name() -> str:
    """Which TPU runtime this process uses."""
    global _RUNTIME
    if _RUNTIME is None:
        chosen = os.environ.get("DIFFLET_TPU_RUNTIME", TORCH_XLA).strip() or TORCH_XLA
        if chosen not in _RUNTIMES:
            raise ValueError(
                f"DIFFLET_TPU_RUNTIME={chosen!r} is not one of {_RUNTIMES}"
            )
        _RUNTIME = chosen
    return _RUNTIME


# ------------------------------------------------------------------ device


def device():
    """This process's TPU device. Replaces ``torch_xla.device()``."""
    if runtime_name() == TORCH_XLA:
        import torch_xla

        return torch_xla.device()

    import torch

    _install_compat()  # idempotent; covers single-process use with no initialize_replica()
    return torch.device("tpu")


def device_kind() -> str:
    """Chip generation (e.g. ``TPU v5 lite``), or ``unknown``.

    Replaces ``xm.xla_device_hw(xm.xla_device())``. Used for capacity
    decisions — v5e has 16 GB HBM per chip against v5p's ~95 GB, which is the
    difference between a model fitting at tp=1 and not.
    """
    try:
        if runtime_name() == TORCH_XLA:
            import torch_xla.core.xla_model as xm

            return str(xm.xla_device_hw(xm.xla_device()) or "unknown")

        import torch

        # Note the string differs between runtimes for the same chip:
        # torch_xla says "TPU v5 lite", torch_tpu says "TPU v5e". Anything
        # matching on it must accept both.
        return str(torch.tpu.get_device_name() or "unknown")
    except Exception:  # noqa: BLE001 — a probe must never break import
        return "unknown"


# -------------------------------------------------------------- topology


def rank() -> int:
    """Global ordinal of this process. Replaces ``xr.global_ordinal()``."""
    if runtime_name() == TORCH_XLA:
        import torch_xla.runtime as xr

        return int(xr.global_ordinal())

    import torch.distributed as dist

    return int(dist.get_rank()) if dist.is_initialized() else 0


def world_size() -> int:
    """Replaces ``xr.world_size()``.

    torch_xla 2.9 removed ``xm.get_ordinal()``/``xm.xrt_world_size()``; the
    replacements live on ``torch_xla.runtime``.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.runtime as xr

        return int(xr.world_size())

    import torch.distributed as dist

    return int(dist.get_world_size()) if dist.is_initialized() else 1


#: Where TorchTPU's tier-3 compilation cache lives. Overridable, because the
#: tier is only worth having when the path outlives the process -- a shared
#: filesystem on a pod, a local disk on one box.
TIER3_CACHE_ENV = "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"
DEFAULT_TIER3_CACHE = ".difflet-cache/torchtpu"


def _enable_compilation_cache() -> None:
    """Point TorchTPU's tier-3 cache somewhere persistent.

    This is the whole reason for the migration and it is **off by default**.
    torch_xla cannot deserialise an executable at all (``UNIMPLEMENTED:
    Deserializing serialized executable not supported``), which is why every
    fresh v5e worker recompiles and why warm start is 268.8 s on LTX-2 against
    trn2's 58.5 s. TorchTPU can, but only once a root path is set: tier-1 is
    per-process, tier-2 is per-host shared memory, and neither survives a
    reboot or reaches a second host.

    Phase 0d measured 37 compilations dropping to 0 across a fresh process with
    this set. The first real-model benchmarks were run without it -- they still
    beat torch_xla on cold start, but on tier-2 dedup between the four workers
    alone.

    Setting a default rather than leaving it unset: an operator who has to know
    an undocumented environment variable to get the headline benefit will not
    get it. The variable is undocumented in the literal sense -- it appears in
    neither TorchTPU's docs nor its public API list, and was found with
    ``strings`` over the shipped binary.
    """
    if os.environ.get(TIER3_CACHE_ENV):
        return
    root = pathlib.Path(
        os.environ.get("DIFFLET_TPU_CACHE_DIR") or DEFAULT_TIER3_CACHE
    ).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return  # read-only or unwritable: run without the tier rather than fail
    os.environ[TIER3_CACHE_ENV] = str(root.resolve())


_COMPAT_INSTALLED = False


def _install_compat() -> None:
    """Work around TorchTPU op gaps that break unmodified diffusers models.

    **group_norm rejects non-contiguous input.** Measured on a minimal repro:
    a channels-last ``[1, 512, 128, 128]`` tensor raises ``view(): cannot
    create a view of shape [1, 32, 262144]`` — the op reshapes with ``view``
    where CPU and CUDA PyTorch accept any strides. ``.contiguous()`` first fixes
    it and matches CPU to 1-cos 2.6e-6. diffusers' ``AutoencoderKL`` feeds
    group_norm such a tensor partway through its decoder, so FLUX.1-dev's VAE
    decode failed on torch_tpu after the whole denoise had succeeded. Qwen-Image
    and Wan were never affected: their VAEs normalise with RMS-style layers.

    The wrapper copies only when the input is on the tpu device *and* not
    contiguous, so the host-side text encoders are untouched and the common
    path costs one ``is_contiguous()`` check. The copy itself is ~16 MB at FLUX's
    1024^2 decode, negligible beside the VAE.

    Patched at ``torch.nn.functional.group_norm`` because ``nn.GroupNorm``
    resolves ``F.group_norm`` at call time, so one attribute covers every model.
    Idempotent. A candidate for an upstream TorchTPU bug report; the repro is
    ``scripts/torchtpu_groupnorm_repro.py``.
    """
    global _COMPAT_INSTALLED
    if _COMPAT_INSTALLED or runtime_name() != TORCH_TPU:
        return

    import functools

    import torch.nn.functional as F

    original = F.group_norm

    @functools.wraps(original)
    def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
        if input.device.type == "tpu" and not input.is_contiguous():
            input = input.contiguous()
        return original(input, num_groups, weight, bias, eps)

    F.group_norm = group_norm
    _COMPAT_INSTALLED = True


#: Host-to-device copies only take TorchTPU's fast path if the whole tensor
#: fits in this premapped buffer. 2 GiB clears the largest on-chip tensor of
#: every supported model (see _size_premapped_buffer) with room to spare.
PREMAPPED_BUFFER_ENV = "TPU_PREMAPPED_BUFFER_SIZE"
DEFAULT_PREMAPPED_BUFFER = 2 * 2 ** 30


def _size_premapped_buffer() -> None:
    """Make host-to-device weight copies take TorchTPU's fast path.

    **This is why torch_tpu loaded weights 20-27x slower than torch_xla.** A copy
    goes at ~12 GB/s only if the *whole tensor* fits in the premapped buffer;
    one that does not falls back to a single-threaded host copy at ~0.2 GB/s.
    It does not chunk through the buffer. Measured with a 256 MiB buffer and
    pinned memory:

        transfer 32-240 MiB (fits)       10.9-12.7 GB/s
        transfer 256 / 300 / 512 MiB      0.20 GB/s

    and with the buffer large enough, faster than torch_xla's 5.6 GB/s:
    13.0 GB/s pinned, 7.0 pageable. Under the default, cpu/wall sat at ~1.1 —
    one core doing the copy by hand — and pinned memory, non_blocking, dtype
    and shape made no difference, because none of them is the bottleneck.

    So what matters is the **largest single tensor**, not the total weight.
    From the checkpoint headers, the largest that goes to the chip:

        Qwen-Image 108 MiB · HunyuanVideo 108 · FLUX.1-dev 126
        LTX-2 216 (its VAE runs on chip) · Wan 2.1 600 (F32 checkpoint)

    That last line is why Wan's load gap was larger than Qwen-Image's despite
    less weight per rank: its tensors are individually bigger, so more of them
    went over the cliff. 2 GiB also covers LTX-2's 1.35 GiB text connector,
    should that ever move to the chip.

    Cost: 2 GiB of premapped host memory per worker, 8 GiB for a tp=4 host of
    188 GB. setdefault, so an operator can trade it down for a model with
    small tensors. The variable is in TorchTPU's public API list as stable;
    nothing there says the default makes weight loading 60x slower.
    """
    os.environ.setdefault(PREMAPPED_BUFFER_ENV, str(DEFAULT_PREMAPPED_BUFFER))


def _enable_fusion() -> None:
    """Turn on TorchTPU's DeferAndFuse eager mode. Measured: it is not optional.

    TorchTPU defaults to Strict eager -- one aten op dispatched at a time. The
    torch_xla path this replaces accumulated a whole denoise step into one XLA
    graph via mark_step(), so running Strict is the difference between one
    fused graph and a few thousand individual dispatches.

    On Qwen-Image at 1024^2, tp=4, measured per DiT step:

        torch_xla (one graph)        0.270 s
        torch_tpu Strict eager       1.004 s     <- the default
        torch_tpu DeferAndFuse       0.357 s

    So Strict costs 647 ms of the 734 ms gap; the 87 ms left over is the
    attention kernel (SDPA here against torch_xla's Pallas flash kernel), which
    is Phase 4's problem and a far smaller one.

    Set here rather than left to the caller: a difflet that is 2.8x slower
    unless an operator remembers an environment variable is not a default worth
    shipping. An operator who sets it explicitly still wins -- setdefault.
    """
    os.environ.setdefault("TORCH_TPU_DEFER_AND_FUSE", "1")


def _free_port() -> int:
    """A port the rendezvous can own, picked the way torchrun picks one."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def prepare_launch(nproc: int) -> None:
    """Set up the environment TPU workers need, **in the parent, before spawn**.

    No-op on torch_xla, which needs nothing from the parent.

    On torch_tpu it is required and easy to miss: ``init_process_group
    ("tpu_dist")`` fails with

        missing required environment variables for distributed training:
        TORCH_TPU_SLICEBUILDER_ADDRESSES, TORCH_TPU_TOPOLOGY; please run the
        program via torchrun or similar tools

    and that advice is misleading — torchrun does not set them either.
    ``torch.tpu.distributed.set_tpu_launch_env()`` does, and its contract is
    that it runs once in the parent before any worker exists. Discovered in
    Phase 0b of docs/plans/2026-10-22-torchtpu-migration.md.

    Also pins ``TORCH_TPU_DEVICES_PER_HOST``; without it the process group
    warns "Node-local store disabled" and every rank publishes to the world
    store individually.
    """
    if runtime_name() == TORCH_XLA:
        return

    import torch

    _enable_fusion()
    _enable_compilation_cache()
    _size_premapped_buffer()

    os.environ.setdefault("TORCH_TPU_DEVICES_PER_HOST", str(nproc))
    torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=nproc)

    # set_tpu_launch_env supplies only the TPU-specific variables
    # (TORCH_TPU_TOPOLOGY, TORCH_TPU_SLICEBUILDER_ADDRESSES). torch.distributed's
    # env:// rendezvous needs its own, which torchrun would normally provide --
    # and difflet does not use torchrun: both the resident worker and the
    # benchmark adapter spawn with mp.Process, so the children inherit whatever
    # is set here and nothing else. setdefault throughout, so a real torchrun
    # launch keeps its own values.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    os.environ.setdefault("WORLD_SIZE", str(nproc))


def initialize_replica(replica_rank: int, replica_count: int) -> None:
    """Give this process its own chip, before anything else touches one.

    Replaces ``torch_xla._internal.pjrt.initialize_multiprocess`` in
    ``serving/engines/resident_worker.py``, which reimplements what
    ``torch_xla.launch`` does internally because launch owns the process
    lifecycle and a resident worker has to outlive a single call.
    """
    if runtime_name() == TORCH_XLA:
        from torch_xla._internal import pjrt

        pjrt.initialize_multiprocess(replica_rank, replica_count)
        return

    import torch.distributed as dist

    # Belt and braces: the parent's prepare_launch() normally sets this and the
    # child inherits it, but a process that comes up without a parent (a single
    # worker, a torchrun launch) would otherwise run Strict eager at 2.8x the
    # cost. setdefault, so an inherited value wins.
    _enable_fusion()
    _enable_compilation_cache()
    _size_premapped_buffer()
    _install_compat()

    # TorchTPU hands each worker its own chip, so there is no set_device_index
    # here: the process sees exactly one addressable device and any index but 0
    # raises. The parent must have called prepare_launch() before spawning, or
    # this fails on missing TORCH_TPU_TOPOLOGY / TORCH_TPU_SLICEBUILDER_ADDRESSES.
    # torch_xla's pjrt.initialize_multiprocess took the rank as an argument;
    # torch.distributed reads it from the environment. setdefault so a torchrun
    # launch, which sets these correctly already, is left alone.
    os.environ.setdefault("RANK", str(replica_rank))
    os.environ.setdefault("LOCAL_RANK", str(replica_rank))
    os.environ.setdefault("WORLD_SIZE", str(replica_count))

    if not dist.is_initialized():
        dist.init_process_group(backend="tpu_dist")


# ----------------------------------------------------------- synchronization


def step() -> None:
    """Flush the pending graph. Replaces ``xm.mark_step()``.

    Dispatches the accumulated work; it does **not** wait for it. Call sites
    that need the result on the host follow it with :func:`wait`.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        return

    # Nothing to do: TorchTPU has no lazy graph to flush. What matters is that
    # DeferAndFuse is on -- see _enable_fusion(), called from prepare_launch().


def wait() -> None:
    """Block until queued device work finishes. Replaces ``xm.wait_device_ops()``."""
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        xm.wait_device_ops()
        return

    import torch

    torch.accelerator.synchronize()


def sync() -> None:
    """``step()`` then ``wait()`` — the pair several call sites already used."""
    step()
    wait()


# ---------------------------------------------------- torch_tpu process groups


#: Replica-group list -> this rank's ProcessGroup. torch_xla takes the group
#: lists per call; torch.distributed wants objects built collectively, so they
#: are built once and cached.
_PROCESS_GROUPS: dict[tuple[tuple[int, ...], ...], object] = {}


def _dist_fn(dist, preferred: str, legacy: str):
    """The current spelling of a collective, falling back to the old one.

    torch 2.14 deprecates ``all_gather_into_tensor`` for ``all_gather_single``
    and ``reduce_scatter_tensor`` for ``reduce_scatter_single``, warning on
    every call. Picking the new name when it exists keeps this code off the
    path that is already being removed — which is the whole reason this
    migration exists — without requiring the newer torch.
    """
    return getattr(dist, preferred, None) or getattr(dist, legacy)


def _process_group(groups: list[list[int]]):
    """The subgroup this rank belongs to within ``groups``.

    **``dist.new_group`` is collective**: every rank must call it for every
    group, in the same order, or the ones that skip a call deadlock the ones
    that make it. That holds here because the lists come from
    ``parallel_mesh.axis_groups()``, which derives a full partition of the
    world from a ``MeshSpec`` every rank computes identically — so each rank
    walks the same keys in the same order and the cache fills in lockstep.
    Anything that starts passing rank-dependent groups breaks that, which is
    why the partition is re-checked here rather than trusted.
    """
    import torch.distributed as dist

    key = tuple(tuple(int(r) for r in g) for g in groups)
    if key in _PROCESS_GROUPS:
        return _PROCESS_GROUPS[key]

    flat = sorted(r for g in key for r in g)
    world = dist.get_world_size()
    if flat != list(range(world)):
        raise ValueError(
            f"replica groups must partition all {world} ranks exactly once; got {groups}"
        )

    me = dist.get_rank()
    mine = None
    for members in key:  # every rank creates every group, in this order
        created = dist.new_group(ranks=list(members))
        if me in members:
            mine = created
    _PROCESS_GROUPS[key] = mine
    return mine


# ------------------------------------------------------------- collectives


def broadcast(tensors: list, *, src: int = 0) -> None:
    """Broadcast a list of device tensors in place from rank ``src``.

    Replaces ``xm.collective_broadcast(payload, root_ordinal=0)``, which every
    ``tpu_application`` uses to fan an encoded prompt out from the encoder rank
    together with a status flag.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        xm.collective_broadcast(tensors, root_ordinal=src)
        return

    import torch.distributed as dist

    for tensor in tensors:  # xm.collective_broadcast takes the list; dist does not
        dist.broadcast(tensor, src=src)


def all_gather(tensor, *, dim: int, groups: list[list[int]]):
    """Replaces ``xm.all_gather(..., groups=, pin_layout=False)``.

    ``pin_layout=False`` is load-bearing: pinned layouts lower to an
    ``mhlo.all_gather`` that the TPU compiler handles differently.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        return xm.all_gather(tensor, dim=dim, groups=groups, pin_layout=False)

    import torch
    import torch.distributed as dist

    pg = _process_group(groups)
    size = len(groups[0])
    source = tensor.contiguous()
    # all_gather_single always stacks along dim 0, in rank order.
    gathered = torch.empty(
        (size * source.shape[0], *source.shape[1:]),
        dtype=source.dtype, device=source.device,
    )
    _dist_fn(dist, "all_gather_single", "all_gather_into_tensor")(
        gathered, source, group=pg
    )
    if dim == 0 or dim == -source.dim():
        return gathered
    return torch.cat(gathered.chunk(size, dim=0), dim=dim)


def all_reduce_sum(tensor, *, groups: list[list[int]]):
    """Replaces ``xm.all_reduce(xm.REDUCE_SUM, ..., groups=, pin_layout=False)``."""
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        return xm.all_reduce(xm.REDUCE_SUM, tensor, groups=groups, pin_layout=False)

    import torch.distributed as dist

    # xm.all_reduce returns a new tensor; dist.all_reduce is in place, so clone
    # first rather than mutating a caller's tensor.
    out = tensor.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=_process_group(groups))
    return out


def reduce_scatter_sum(tensor, *, dim: int, shard_count: int, groups: list[list[int]]):
    """Replaces ``xm.reduce_scatter(xm.REDUCE_SUM, ..., scale=1.0, ...)``."""
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        return xm.reduce_scatter(
            xm.REDUCE_SUM,
            tensor,
            scale=1.0,
            scatter_dim=dim,
            shard_count=shard_count,
            groups=groups,
            pin_layout=False,
        )

    import torch
    import torch.distributed as dist

    pg = _process_group(groups)
    # reduce_scatter_single always scatters along dim 0; bring the target axis
    # to the front and put it back afterwards.
    axis = dim % tensor.dim()
    source = tensor if axis == 0 else tensor.movedim(axis, 0)
    source = source.contiguous()
    if source.shape[0] % shard_count:
        raise ValueError(
            f"cannot reduce-scatter dimension {axis} of size {source.shape[0]} "
            f"across {shard_count} shards"
        )
    out = torch.empty(
        (source.shape[0] // shard_count, *source.shape[1:]),
        dtype=source.dtype, device=source.device,
    )
    _dist_fn(dist, "reduce_scatter_single", "reduce_scatter_tensor")(
        out, source, op=dist.ReduceOp.SUM, group=pg
    )
    return out if axis == 0 else out.movedim(0, axis)


# ------------------------------------------------------------- compilation

#: Opt-in, while it is being measured: run a module through TorchTPU's
#: whole-graph mode instead of DeferAndFuse eager.
COMPILE_ENV = "DIFFLET_TPU_COMPILE"


def maybe_compile(module):
    """``torch.compile(module, backend="tpu")`` under torch_tpu when opted in.

    The working hypothesis for torch_tpu's remaining per-step gap is fusion
    completeness. torch_xla compiles a whole denoise step into one XLA graph;
    DeferAndFuse is heuristic, and a custom kernel cuts it further (a jax_op
    attention turned 0.333 s into 0.566 s per step on Qwen-Image). TorchTPU's
    compiled mode lowers one FX graph to one StableHLO program, the nearest
    thing it has to torch_xla's model. This exists to test that.

    Unchanged on torch_xla, where the step is already one lazy graph, and
    unchanged unless ``DIFFLET_TPU_COMPILE`` is set. TorchTPU sets
    ``dynamic=False`` itself; callers must keep shapes static and pass varying
    scalars as device tensors, which the TPU loops here already do.
    """
    if runtime_name() != TORCH_TPU:
        return module
    if os.environ.get(COMPILE_ENV, "0").strip().lower() not in ("1", "true", "yes"):
        return module

    import torch

    print("[difflet tpu] torch.compile(backend='tpu') applied to %s"
          % type(module).__name__, flush=True)
    return torch.compile(module, backend="tpu")


# ------------------------------------------------------------------ memory


def memory_info(dev=None) -> dict:
    """HBM counters for ``dev``. Replaces ``xm.get_memory_info(device)``.

    Feeds the "resident HBM / chip" column in ``benchmark/v5e/RESULTS.md``,
    which is how a model is judged to fit: a v5e chip has 16 GB against a
    v5p's ~95 GB. Returns the runtime's own dict, so callers keep reading
    ``peak_bytes_used`` / ``bytes_used`` and tolerate either being absent.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        return dict(xm.get_memory_info(dev if dev is not None else device()))

    import torch

    # Shaped like torch_xla's dict so callers (benchmark/adapters/tpu.py) keep
    # reading peak_bytes_used / bytes_used unchanged.
    #
    # **Use the device counter, not the caching allocator.** torch_xla's
    # peak_bytes_used is what the chip reports; torch.accelerator's
    # memory_allocated / max_memory_allocated are the PyTorch allocator's view,
    # which reads ~25% higher (it keeps freed blocks) and does not fall back to
    # zero when tensors are released — measured 2.0 GB device against 2.5 GB
    # allocator, and 0.001 against 1.007 after freeing. Reporting the allocator
    # under torch_xla's key names made benchmark/v5e compare two different
    # quantities and look like a 41% regression that did not exist.
    free, total = torch.accelerator.get_memory_info()
    device_used = int(total) - int(free)
    return {
        "bytes_used": device_used,
        # The device counter is an instantaneous reading, so this is a floor on
        # the true peak rather than the peak itself; the allocator's peak is
        # carried alongside for anyone who wants the allocator's story.
        "peak_bytes_used": device_used,
        "allocator_bytes_used": int(torch.accelerator.memory_allocated()),
        "allocator_peak_bytes_used": int(torch.accelerator.max_memory_allocated()),
        "bytes_limit": int(total),
        "bytes_free": int(free),
    }


# --------------------------------------------------------------- precision


#: XLA's default matmul precision on TPU is NOT fp32: the MXU multiplies fp32
#: inputs at reduced precision unless told otherwise, measured at ~1e-2
#: absolute error per matmul on v5e against ~2e-6 at "highest". Compounded
#: through a DiT that makes the parity target unreachable, and the failure is
#: silent — every shape is right and nothing errors.
#:
#: Phase 0c measured the TorchTPU side: its `medium` default holds cosine far
#: inside the 0.9995 bar on a 4-block FLUX proxy at 2.42x the speed of
#: `highest`, so the Phase 2 default is expected to change. Until a real model
#: confirms it in Phase 4, this stays where torch_xla needs it.
DEFAULT_MATMUL_PRECISION = "highest"

#: Operator override for the above. The default does not change: moving to
#: "medium" is a per-model decision for Phase 4, gated on parity against the
#: diffusers oracle. This exists so that decision can be measured.
MATMUL_PRECISION_ENV = "DIFFLET_TPU_MATMUL_PRECISION"

#: The two runtimes name the same three settings differently. torch_xla's
#: ``set_mat_mul_precision`` takes 'default' / 'high' / 'highest'; torch's
#: ``set_float32_matmul_precision``, which TorchTPU honours, takes 'medium' /
#: 'high' / 'highest'. 'default' and 'medium' are the same thing: one bf16 MXU
#: pass. Passing 'medium' to torch_xla raised, which is how this was found.
#:
#: Note what the setting governs: **fp32** matmuls only. A DiT that runs in
#: bf16 is untouched by it — measured on Qwen-Image under torch_tpu, 'medium'
#: against 'highest' moved per-step by 1% (0.330 s vs 0.333 s).
_PRECISION_NAMES = {
    TORCH_XLA: ("default", "high", "highest"),
    TORCH_TPU: ("medium", "high", "highest"),
}
_PRECISION_ALIASES = {
    TORCH_XLA: {"medium": "default"},
    TORCH_TPU: {"default": "medium"},
}


def _precision_for(runtime: str, precision: str) -> str:
    name = _PRECISION_ALIASES[runtime].get(precision, precision)
    if name not in _PRECISION_NAMES[runtime]:
        accepted = sorted(set(_PRECISION_NAMES[runtime]) | set(_PRECISION_ALIASES[runtime]))
        raise ValueError(f"matmul precision {precision!r} is not one of {accepted}")
    return name


def set_matmul_precision(precision: str | None = None) -> str:
    """Set matmul precision process-wide, returning the previous value.

    Deliberately explicit rather than done at import time: it is process-global
    state, and silently mutating it on import would be worse than the bug it
    prevents.

    Resolution order: an explicit argument, then ``DIFFLET_TPU_MATMUL_PRECISION``,
    then ``DEFAULT_MATMUL_PRECISION``.
    """
    if precision is None:
        precision = os.environ.get(MATMUL_PRECISION_ENV) or DEFAULT_MATMUL_PRECISION
    precision = _precision_for(runtime_name(), precision)
    if runtime_name() == TORCH_XLA:
        import torch_xla.backends as backends

        previous = backends.get_mat_mul_precision()
        backends.set_mat_mul_precision(precision)
        return previous

    import torch

    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    return previous


__all__ = [
    "DEFAULT_MATMUL_PRECISION",
    "DEFAULT_PREMAPPED_BUFFER",
    "DEFAULT_TIER3_CACHE",
    "TORCH_TPU",
    "TORCH_XLA",
    "all_gather",
    "all_reduce_sum",
    "broadcast",
    "device",
    "device_kind",
    "initialize_replica",
    "maybe_compile",
    "memory_info",
    "prepare_launch",
    "rank",
    "reduce_scatter_sum",
    "runtime_name",
    "set_matmul_precision",
    "step",
    "sync",
    "wait",
    "world_size",
]
