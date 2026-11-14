"""The one module on the TPU path that talks to the TPU runtime: TorchTPU.

Every TPU call site routes through here -- device, topology, collectives,
compilation, memory, precision -- so the rest of the backend never imports
``torch_tpu`` or ``torch.tpu`` directly.

The backend ran on ``torch_xla`` until the migration in
docs/plans/2026-10-22-torchtpu-migration.md; ``torch_xla`` is deprecated (its
repository is archived 2026-11-15) and its code path is gone. The numbers it
produced stay in ``benchmark/v5e/`` as a reference only: this code cannot
reproduce them.

Imports stay inside the functions: the TPU backend has to be importable on a
host with no TPU runtime at all (the CPU backend, unit tests on a laptop,
``difflet plan``).
"""

from __future__ import annotations

import contextlib
import os
import pathlib

# ------------------------------------------------------------------ device


def device():
    """This process's TPU device."""
    import torch

    _install_compat()  # idempotent; covers single-process use with no initialize_replica()
    return torch.device("tpu")


def device_kind() -> str:
    """Chip generation (e.g. ``TPU v5 lite``), or ``unknown``.

    Used for capacity
    decisions — v5e has 16 GB HBM per chip against v5p's ~95 GB, which is the
    difference between a model fitting at tp=1 and not.
    """
    try:
        import torch

        return str(torch.tpu.get_device_name() or "unknown")
    except Exception:  # noqa: BLE001 — a probe must never break import
        return "unknown"


# -------------------------------------------------------------- topology


def rank() -> int:
    """Global ordinal of this process."""
    import torch.distributed as dist

    return int(dist.get_rank()) if dist.is_initialized() else 0


def world_size() -> int:
    """Number of TPU worker processes (1 without ``torch.distributed``)."""
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
    if _COMPAT_INSTALLED:
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

    Required and easy to miss: ``init_process_group
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

    Called by ``serving/engines/resident_worker.py`` and the benchmark adapter,
    which spawn their workers with ``mp.Process`` (a resident worker has to
    outlive a single call) after the parent ran :func:`prepare_launch`.
    """
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
    # torch.distributed reads the rank from the environment. setdefault so a torchrun
    # launch, which sets these correctly already, is left alone.
    os.environ.setdefault("RANK", str(replica_rank))
    os.environ.setdefault("LOCAL_RANK", str(replica_rank))
    os.environ.setdefault("WORLD_SIZE", str(replica_count))

    if not dist.is_initialized():
        dist.init_process_group(backend="tpu_dist")


# ----------------------------------------------------------- synchronization


def wait() -> None:
    """Block until queued device work finishes."""
    import torch

    torch.accelerator.synchronize()


# ---------------------------------------------------- torch_tpu process groups


#: Replica-group list -> this rank's ProcessGroup. Callers pass group lists
#: (``parallel_mesh.axis_groups()``); torch.distributed wants objects built
#: collectively, so they are built once and cached.
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

    Every ``tpu_application`` uses it to fan an encoded prompt out from the
    encoder rank together with a status flag.
    """
    import torch.distributed as dist

    for tensor in tensors:
        dist.broadcast(tensor, src=src)


def all_gather(tensor, *, dim: int, groups: list[list[int]]):
    """All-gather ``tensor`` along ``dim`` within this rank's group of ``groups``."""
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
    """Sum ``tensor`` within this rank's group of ``groups``; returns a new tensor."""
    import torch.distributed as dist

    # dist.all_reduce is in place: clone first rather than mutating a caller's
    # tensor.
    out = tensor.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=_process_group(groups))
    return out


def reduce_scatter_sum(tensor, *, dim: int, shard_count: int, groups: list[list[int]]):
    """Sum ``tensor`` within this rank's group and keep this rank's ``1/shard_count`` along ``dim``."""
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


def maybe_compile(module, *, default: bool = False):
    """``torch.compile(module, backend="tpu")``, per model.

    ``default`` is the call site's choice for its model; ``DIFFLET_TPU_COMPILE``
    overrides it either way (``1`` forces on, ``0`` forces off). On by default
    only where it has been measured — Qwen-Image, Wan, FLUX.1-dev and LTX-2 — and off
    for anything that has not run under it.

    Why compile: DeferAndFuse fuses heuristically and left torch_tpu 21% behind
    the old torch_xla backend (one lazy graph per step) on Qwen-Image. Compiled,
    each of the three is one graph (zero graph breaks) and beat it per step —
    Qwen-Image −15%, Wan −41%, FLUX −3% — with output closer to eager (49.3 dB
    PSNR) than the runtime switch itself left it (44.1 dB).

    There is no ahead-of-time artifact: the first call traces (Dynamo) and
    compiles (XLA), and the XLA executables persist in TorchTPU's tier-3
    compilation cache (:func:`_enable_compilation_cache`), so a later process
    skips the XLA compile but still re-traces.

    The cost: the compiled programs are cached across processes, but each new
    process still pays +24-50 s on its first call (most likely Dynamo
    re-tracing; XLA compiles zero times). Break-even is 6-26 requests, so it
    belongs in a resident server — the only TPU entry point today, whose
    startup smoke request absorbs it before ``/ready`` — and a one-shot run
    should set ``DIFFLET_TPU_COMPILE=0``.

    TorchTPU sets ``dynamic=False`` itself; shapes must stay static and varying scalars
    must be device tensors, which the TPU loops here already ensure.
    """
    raw = os.environ.get(COMPILE_ENV, "").strip().lower()
    if raw in ("1", "true", "yes"):
        enabled = True
    elif raw in ("0", "false", "no"):
        enabled = False
    else:
        enabled = default
    if not enabled:
        return module

    import torch

    from difflet.backends.tpu.ops_impl import attention, collectives

    # Build the opt-in Pallas kernel now, outside the compiled region. Built
    # lazily inside it, Dynamo traces into jax_op's registration and
    # infer_schema and graph-breaks 14 times per rank on the first call
    # (2026-10-27); built here, the kernel is one op in the compiled graph.
    attention._flash_kernel()
    collectives.warm_process_groups()
    print("[difflet tpu] torch.compile(backend='tpu') applied to %s"
          % type(module).__name__, flush=True)
    return torch.compile(module, backend="tpu")


DEVICE_VAE_ENV = "DIFFLET_TPU_DEVICE_VAE"
DEVICE_VAE_DTYPE_ENV = "DIFFLET_TPU_DEVICE_VAE_DTYPE"

#: DiT round trips this process has made for a chip decode (diagnostics:
#: benchmark/adapters/tpu.py prints it per request).
VAE_ROOM_ROUND_TRIPS = 0


def maybe_device_vae(vae):
    """Move a video VAE onto this replica's chip when ``DIFFLET_TPU_DEVICE_VAE=1``.

    Wan and HunyuanVideo decode on the host by default -- a leftover of the
    torch_xla port (``AutoencoderKLWan`` raised a negative-index error there, and
    HunyuanVideo was ported host-first), not an HBM limit. Both decode on a chip
    in fp32, as on the host (bf16 is where Trainium's Wan device VAE lost its quality, #117).
    Only the primary replica decodes, so only it holds the VAE; every other
    replica keeps the host copy it never uses.

    A chip VAE needs :func:`vae_decode_room` around every decode: with the DiT
    resident there is no room for it (Wan asks for 4.72 GB with 4.66 free,
    HunyuanVideo 5.65 with 3.78), and a tiled decode squeezed into that room
    stalled 35+ min in the real pipeline (2026-10-28). Measured end to end with
    the room made (v5e-4, tp=4):

        Wan           decode stage 20.8 s -> 5.5 s, resident request 28.8 -> 13.4 s
        HunyuanVideo  decode stage 169 s -> 20.5 s, resident request ~193 -> 46.3 s

    with output equal to the host decode's to 6-7 digits on real latents.
    """
    if os.environ.get(DEVICE_VAE_ENV, "").strip().lower() not in ("1", "true", "yes"):
        return vae
    if int(os.environ.get("DIFFLET_REPLICA_RANK", "0")) != 0:
        return vae
    # fp32 by default. In fp32 every conv runs at HIGHEST precision (several bf16
    # MXU passes); DIFFLET_TPU_DEVICE_VAE_DTYPE=bf16 trades that for speed --
    # 18.2 -> 13.0 s (HunyuanVideo) and 1.44 -> 1.06 s (Wan) per decode on
    # random latents, 54.9 / 56.8 dB PSNR against the host fp32 decode
    # (vae_device_probe, 2026-10-27). Callers cast the latents to the VAE's dtype.
    dtype_name = os.environ.get(DEVICE_VAE_DTYPE_ENV, "fp32").strip().lower()
    import torch

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(dtype_name)
    if dtype is None:
        raise ValueError("%s must be fp32 or bf16, got %r" % (DEVICE_VAE_DTYPE_ENV, dtype_name))
    print("[difflet tpu] %s decodes on the chip (%s)" % (type(vae).__name__, dtype_name), flush=True)
    return vae.to(device(), dtype)


@contextlib.contextmanager
def vae_decode_room(vae, modules):
    """Around a chip VAE decode: move ``modules`` (the DiT) to the host, then back.

    A no-op when ``vae`` is on the host -- every backend but a TPU primary with
    ``DIFFLET_TPU_DEVICE_VAE=1`` -- so a caller can wrap its decode
    unconditionally. The decode must finish inside the block (read its output
    to the host there): under DeferAndFuse it would otherwise run after the DiT
    is back.

    The round trip costs ~3.0 s for Wan's shard and ~4.7 s for HunyuanVideo's
    (the very first one ~22 s), and the compiled DiT survives it: the same
    device, dtype and shapes satisfy its guards, so the next denoise runs at
    the same step time with no recompile. Safe against the next request because
    the resident worker runs one request at a time, decode included
    (``_run_lock``), and the ``finally`` restores the DiT if the decode raises.
    """
    global VAE_ROOM_ROUND_TRIPS
    modules = list(modules or ())
    if not modules or next(vae.parameters()).device.type == "cpu":
        yield
        return
    import time

    def free_gb():
        wait()
        return memory_info()["bytes_free"] / 2**30

    if _peer_stash_enabled():
        before = free_gb()
        mark = time.monotonic()
        groups = _stash_groups(modules)
        if _stash_preflight(groups):
            # Phase 1 copies out and touches no weight; only after every group
            # is on the peers does phase 2 drop the primary's copy. A failure
            # in phase 1 leaves the DiT whole.
            parts = _stash_out(groups, primary=True)
            _stash_commit(groups)
            print("[difflet tpu] DiT -> peers for the VAE decode in %.2f s (free %.2f -> %.2f GB)"
                  % (time.monotonic() - mark, before, free_gb()), flush=True)
            try:
                yield
            finally:
                mark = time.monotonic()
                _stash_back(groups, parts, primary=True)
                VAE_ROOM_ROUND_TRIPS += 1
                print("[difflet tpu] DiT <- peers in %.2f s (free %.2f GB)"
                      % (time.monotonic() - mark, free_gb()), flush=True)
            return
        print("[difflet tpu] DiT peer stash skipped for this decode (not enough free HBM "
              "on some replica); host round trip instead", flush=True)

    before = free_gb()
    mark = time.monotonic()
    for module in modules:
        module.to("cpu")
    print("[difflet tpu] DiT -> host for the VAE decode in %.2f s (free %.2f -> %.2f GB)"
          % (time.monotonic() - mark, before, free_gb()), flush=True)
    try:
        yield
    finally:
        mark = time.monotonic()
        target = device()
        for module in modules:
            module.to(target)
        VAE_ROOM_ROUND_TRIPS += 1
        print("[difflet tpu] DiT -> chip in %.2f s (free %.2f GB)"
              % (time.monotonic() - mark, free_gb()), flush=True)


VAE_PEER_STASH_ENV = "DIFFLET_TPU_VAE_PEER_STASH"

#: Bytes per reduce-scatter group. One group's buffers are the stash's working
#: memory on every chip, so this is kept small: at 512 MiB HunyuanVideo's peers
#: (4.4 GiB free, 2.7 GiB of it taken by their share) could not load the
#: all-gather program (2026-10-28).
_STASH_GROUP_BYTES = 128 * 2**20

#: HBM an all-gather program of one group reserves, as a multiple of the
#: group: measured 1.72 GiB for a 512 MiB group (3.4x) on HunyuanVideo.
_STASH_GROUP_RESERVE = 3.5


def _peer_stash_enabled() -> bool:
    """``DIFFLET_TPU_VAE_PEER_STASH=1`` (with the chip VAE): park the primary's
    DiT on the other chips over ICI instead of the host.

    The host round trip is 3.2 s per Wan request (4-5 GB/s per tensor, ~7 GB/s
    for one contiguous buffer and 0.5-1 GB/s at 2-3 GiB), while the all-reduce
    path between chips moved 1-3 GiB at ~51 GB/s and every peer keeps 8.73 GiB
    free in the serving state (benchmark/tpu_profile/peer_stash_probe.py,
    2026-10-28). Every replica must agree on it -- it is collective -- so it is
    read from the environment the replicas share.
    """
    if os.environ.get(DEVICE_VAE_ENV, "").strip().lower() not in ("1", "true", "yes"):
        return False
    import torch.distributed as dist

    if not (os.environ.get(VAE_PEER_STASH_ENV, "").strip().lower() in ("1", "true", "yes")
            and dist.is_initialized() and dist.get_world_size() > 1):
        return False
    return _peer_stash_over_ici()


#: Cached verdict of :func:`_peer_stash_over_ici` -- it is collective, so it is
#: decided once, at the first decode, when every replica reaches it together.
_PEER_STASH_ICI = None


def _peer_stash_over_ici() -> bool:
    """Only stash over ICI: every replica's chip in one TPU slice.

    Chips in a slice are wired to each other by ICI (~51 GB/s measured here);
    between slices traffic crosses the data-center network, which can be slower
    than the host round trip the stash replaces. JAX in every torch_tpu worker
    sees all devices, so each replica can check that they share one
    ``slice_index`` and that there are as many as ranks. The verdicts are then
    all-reduced (MIN): a stash that one replica skips while the others enter it
    would leave their collectives waiting forever. Off -> the host round trip,
    with the reason logged once.
    """
    global _PEER_STASH_ICI
    if _PEER_STASH_ICI is not None:
        return _PEER_STASH_ICI
    import torch
    import torch.distributed as dist

    world = dist.get_world_size()
    try:
        import jax

        devices = jax.devices()
        slices = sorted({int(getattr(d, "slice_index", 0) or 0) for d in devices})
        if len(slices) != 1:
            ok, why = False, "devices span %d TPU slices (DCN between them)" % len(slices)
        elif len(devices) < world:
            ok, why = False, "%d ranks but JAX sees %d devices" % (world, len(devices))
        else:
            ok, why = True, "%d chips in one TPU slice" % len(devices)
    except Exception as exc:  # noqa: BLE001 -- cannot tell, so do not stash
        ok, why = False, "topology unknown (%r)" % (exc,)
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device())
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    agreed = bool(int(flag.cpu()[0]))  # read back: the all-reduce must really run
    if ok and not agreed:
        why = "another replica cannot use ICI"
    _PEER_STASH_ICI = agreed
    if int(os.environ.get("DIFFLET_REPLICA_RANK", "0")) == 0:
        print("[difflet tpu] DiT peer stash %s: %s"
              % ("on" if agreed else "off, host round trip instead", why), flush=True)
    return agreed


def _stash_groups(modules) -> list:
    """The modules' parameters and buffers in a fixed order, cut into same-dtype
    groups of about ``_STASH_GROUP_BYTES``. Every replica builds the same list
    from its own shard (tp shards share shapes), which is what lines the
    collectives up."""
    seen, tensors = set(), []
    for module in modules:
        for t in list(module.parameters()) + list(module.buffers()):
            if id(t) not in seen:
                seen.add(id(t))
                tensors.append(t)
    groups, current, size = [], [], 0
    for t in tensors:
        nbytes = t.numel() * t.element_size()
        if current and (t.dtype != current[0][0].dtype or size + nbytes > _STASH_GROUP_BYTES):
            groups.append(current)
            current, size = [], 0
        current.append((t, tuple(t.shape)))
        size += nbytes
    if current:
        groups.append(current)
    return groups


def _read_one(t) -> float:
    # Reads one element so the collective that produced ``t`` really runs: under
    # lazy execution an unread result -- or a source buffer nothing wrote -- was
    # dropped while the other ranks waited (four hangs, 2026-10-28).
    return float(t.view(-1)[-1].cpu())


def _stash_out(groups, *, primary: bool) -> list:
    """Reduce-scatter each group from the primary (zeros elsewhere) so every chip
    holds a quarter. Leaves the weights alone: :func:`_stash_commit` drops the
    primary's copy once every group is out."""
    import torch
    import torch.distributed as dist

    world = dist.get_world_size()
    dev = device()
    parts = []
    for group in groups:
        dtype = group[0][0].dtype
        n = sum(t.numel() for t, _ in group)
        padded = n + (-n) % world
        if primary:
            flat = torch.cat([t.data.reshape(-1) for t, _ in group])
            if padded != n:
                flat = torch.cat([flat, torch.zeros(padded - n, dtype=dtype, device=dev)])
        else:
            flat = torch.zeros(padded, dtype=dtype, device=dev)
        _read_one(flat)
        part = torch.empty(padded // world, dtype=dtype, device=dev)
        dist.reduce_scatter_tensor(part, flat)
        _read_one(part)
        del flat
        parts.append(part)
    wait()
    return parts


def _stash_commit(groups) -> None:
    """The primary drops its copy of every stashed tensor (phase 2)."""
    import torch

    dev = device()
    for group in groups:
        for t, _ in group:
            t.data = torch.empty(0, dtype=t.dtype, device=dev)
    wait()


#: Head room kept free on every chip beyond what the stash itself needs.
_STASH_MARGIN_BYTES = 256 * 2**20


def _stash_preflight(groups) -> bool:
    """Does every replica have room for its quarter plus one group's buffer?

    Each replica checks its own free HBM, then all agree (all-reduce MIN): one
    replica short means none stashes this time -- the primary takes the host
    round trip and the others return -- so no collective is left half-entered.
    """
    import torch
    import torch.distributed as dist

    world = dist.get_world_size()
    padded = [sum(t.numel() for t, _ in g) + (-sum(t.numel() for t, _ in g)) % world for g in groups]
    sizes = [n * g[0][0].element_size() for n, g in zip(padded, groups)]
    need = sum(sizes) // world + int(_STASH_GROUP_RESERVE * max(sizes)) + _STASH_MARGIN_BYTES
    wait()
    ok = memory_info()["bytes_free"] >= need
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device())
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(int(flag.cpu()[0]))


def _stash_back(groups, parts, *, primary: bool) -> None:
    """All-gather each group back; the primary refills its tensors in place."""
    import torch
    import torch.distributed as dist

    world = dist.get_world_size()
    dev = device()
    for group, part in zip(groups, parts):
        full = torch.empty(part.numel() * world, dtype=part.dtype, device=dev)
        dist.all_gather_into_tensor(full, part)
        _read_one(full)
        if primary:
            offset = 0
            for t, shape in group:
                k = 1
                for s in shape:
                    k *= s
                t.data = full[offset:offset + k].view(shape).clone()
                offset += k
            _read_one(group[-1][0].data)
        del full
    del part
    parts.clear()
    wait()


def vae_peer_hold(modules) -> None:
    """A non-primary replica's side of the peer stash: take a quarter of the
    primary's DiT, then give it back -- blocking in the all-gather until the
    primary's decode is done. A no-op unless the stash is on."""
    modules = list(modules or ())
    if not modules or not _peer_stash_enabled():
        return
    groups = _stash_groups(modules)
    if not _stash_preflight(groups):
        return
    trace = os.environ.get("DIFFLET_TPU_HBM_TRACE", "").strip() == "1"

    def free_gb():
        wait()
        return memory_info()["bytes_free"] / 2**30

    before = free_gb() if trace else 0.0
    parts = _stash_out(groups, primary=False)
    held = free_gb() if trace else 0.0
    _stash_back(groups, parts, primary=False)
    del parts
    if trace:
        # Per replica, so a share that is not released shows up as a free
        # figure that keeps falling request after request (measured steady at
        # 8.72 -> 6.97 -> 8.72 GB over five Wan requests, 2026-10-28).
        print("[difflet tpu] replica %s held a DiT share: free %.2f -> %.2f -> %.2f GB"
              % (os.environ.get("DIFFLET_REPLICA_RANK", "?"), before, held, free_gb()),
              flush=True)


# ------------------------------------------------------------------ memory


def memory_info(dev=None) -> dict:
    """HBM counters for ``dev``: ``bytes_used`` / ``peak_bytes_used`` / ``bytes_free`` ...

    Feeds the "resident HBM / chip" column of the benchmark reports, which is
    how a model is judged to fit: a v5e chip has 16 GB against a v5p's ~95 GB.
    The key names are the ones ``benchmark/adapters/tpu.py`` reads (inherited
    from torch_xla's dict, so the recorded v5e numbers stay comparable).
    """
    import torch

    # **Use the device counter, not the caching allocator.** It is what the
    # chip reports; torch.accelerator's
    # memory_allocated / max_memory_allocated are the PyTorch allocator's view,
    # which reads ~25% higher (it keeps freed blocks) and does not fall back to
    # zero when tensors are released — measured 2.0 GB device against 2.5 GB
    # allocator, and 0.001 against 1.007 after freeing. Reporting the allocator
    # made benchmark/v5e compare two different quantities and look like a 41%
    # regression that did not exist.
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
#: Phase 0c measured TorchTPU's `medium` holding cosine far inside the 0.9995
#: bar on a 4-block FLUX proxy at 2.42x the speed of `highest`; moving the
#: default is a per-model decision gated on parity against the diffusers oracle.
DEFAULT_MATMUL_PRECISION = "highest"

#: Operator override for the above, so that decision can be measured.
MATMUL_PRECISION_ENV = "DIFFLET_TPU_MATMUL_PRECISION"

#: ``torch.set_float32_matmul_precision`` names, which TorchTPU honours.
#: 'default' (torch_xla's name for one bf16 MXU pass) is accepted as 'medium'.
#:
#: Note what the setting governs: **fp32** matmuls only. A DiT that runs in
#: bf16 is untouched by it — measured on Qwen-Image, 'medium' against
#: 'highest' moved per-step by 1% (0.330 s vs 0.333 s).
_PRECISION_NAMES = ("medium", "high", "highest")
_PRECISION_ALIASES = {"default": "medium"}


def _precision_for(precision: str) -> str:
    name = _PRECISION_ALIASES.get(precision, precision)
    if name not in _PRECISION_NAMES:
        accepted = sorted(set(_PRECISION_NAMES) | set(_PRECISION_ALIASES))
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
    precision = _precision_for(precision)
    import torch

    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    return previous


__all__ = [
    "DEFAULT_MATMUL_PRECISION",
    "DEFAULT_PREMAPPED_BUFFER",
    "DEFAULT_TIER3_CACHE",
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
    "set_matmul_precision",
    "wait",
    "world_size",
]
