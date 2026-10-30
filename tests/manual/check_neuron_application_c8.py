"""Manual check (C8): the non-AoT application base on 4 NeuronCores.

Run one invocation at a time, only when no other process holds the NeuronCores (``neuron-ls``
shows no process and ``ps -eo pid,args | grep -E 'torchrun|tests/manual/check_neuron'`` prints
nothing but the grep). From the repository root, with ``WORK`` any scratch directory:

    DIFFLET_BACKEND=neuron DIFFLET_DISABLE_PREWARM=1 NEURON_RT_NUM_CORES=4 PYTHONPATH=. \
      timeout 1800 torchrun --standalone --nproc-per-node 4 \
      tests/manual/check_neuron_application_c8.py --exec-mode eager --work-dir "$WORK"

then ``--exec-mode compile``, then ``--exec-mode eager --inject-failure example_inputs``,
``--inject-failure build_module`` and ``--inject-failure forward``. NEFFs go to the Difflet
compile cache (``DIFFLET_COMPILE_CACHE``/neuron/neff, set by ``prepare_runtime``).

Happy path:

* broadcast: eager ``broadcast_tensor`` of every supported dtype and shape on the device; int64
  values at the int32 limits arrive exact, and int64 values outside the int32 range are
  rejected (rank 0 raises ValueError, the others RankFailureError) with the ranks still in step;
* lifecycle, per dtype (float32, bfloat16): ToyApplication runs build on meta -> per-rank shard
  load onto neuron -> eager or per-block compiled -> warm-up at the target shape, and a forward
  on an input broadcast from rank 0 must match the CPU TP1 reference, be identical on every rank
  and record zero CPU fallbacks. In compile mode every block must also be compiled through
  ``compile_blocks(backend="neuron")`` with the default options (the neuron backend received
  ``fallback_execution: False``), the warm-up forward must have compiled or loaded a NEFF
  (``CompilationCache.TotalCompilations + PersistentHits >= 1``) and a second forward at the
  warm-up shape must compile and load none;
* unwarmed shape (compile mode, bfloat16): a forward at another sequence length is recorded in
  ``unwarmed_shapes``, compiles or loads a NEFF and still matches its CPU reference.

Failure injection (eager): with ``example_inputs`` or ``build_module`` rank 0 raises before a
collective, and every rank must return promptly, rank 0 with the injected error and ranks 1-3
with RankFailureError naming rank 0 (exit 0). With ``forward`` rank 2 raises inside block 0 of
the warm-up forward, between its two all-reduces: it prints its PASS line and exits with code 3
without a status collective, and torchrun must stop ranks 0, 1 and 3 (blocked in the MLP
all-reduce) well before the timeout; torchrun then exits 1 naming rank 2 (exitcode 3) as the
root cause.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist

from difflet.backends.neuron.core.distributed import (
    RankFailureError,
    broadcast_tensor,
    rank0_call,
    world_info,
)
from difflet.backends.neuron.runtime import track_fallbacks
from difflet.backends.registry import get_backend
from difflet.pipeline.parallel_config import DiffletParallelConfig
from tests.unit.backends._neuron_toy import (
    FORWARD_FAILURE_RANK,
    INJECT_FAILURES,
    ToyApplication,
    toy_blocks_input,
    toy_blocks_reference,
    write_toy_checkpoint,
)

N_BLOCKS, DIM, HIDDEN, BATCH, SEQ = 4, 128, 512, 1, 64
SEQ_UNWARMED = 128
FP32_REL_TOL = 1e-3  # a wrong shard is O(1); fp32 on the device tracks CPU to ~1e-6
BF16_ERR_FACTOR = 3.0  # device bf16 TP4 vs CPU bf16 TP1 mean error, both against CPU fp32
BROADCAST_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64)
BROADCAST_SHAPES = ((3, 5), (), (0, 4))
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
FORWARD_FAILURE_EXIT = 3
DEVICE = torch.device("neuron")


def report(rank: int, ok: bool, line: str) -> None:
    if rank == 0 or not ok:
        suffix = "" if rank == 0 else f" [rank {rank}]"
        print(f"{'PASS' if ok else 'FAIL'} {line}{suffix}", flush=True)


def neff_counters() -> tuple[int, int]:
    """(TotalCompilations, PersistentHits) of the runtime's compilation cache."""
    import torch_neuronx.metrics as metrics

    return (
        metrics.get_counter_value("CompilationCache.TotalCompilations") or 0,
        metrics.get_counter_value("CompilationCache.PersistentHits") or 0,
    )


class CountingToyApplication(ToyApplication):
    """ToyApplication that records the NEFF counters across every top-level forward.

    The top-level ToyBlocksModel stays eager and only calls its blocks, so in compile mode the
    counters moved inside one forward belong to the compiled block graphs alone. The neuron
    backend submits a graph's NEFF compile asynchronously, so each hook synchronizes first:
    earlier work lands before the window opens, this forward's before it closes.
    """

    def build_module(self) -> torch.nn.Module:
        model = super().build_module()
        self.forward_neffs: list[tuple[int, int]] = []
        model.register_forward_pre_hook(self._before_forward)
        model.register_forward_hook(self._after_forward)
        return model

    def _before_forward(self, module, args) -> None:
        torch.neuron.synchronize()
        self._neffs_before = neff_counters()

    def _after_forward(self, module, args, output) -> None:
        torch.neuron.synchronize()
        after = neff_counters()
        self.forward_neffs.append(tuple(a - b for a, b in zip(after, self._neffs_before)))


def block_backend(block: torch.nn.Module) -> tuple[str, dict | None]:
    """(backend name, options dict) that Dynamo hands the backend for a block compiled in place.

    Reads torch 2.14 internals (compiled fn closure -> OptimizeContext -> ConvertFrameAssert ->
    WrapBackendDebug) rather than wrapping the backend, so the block keeps the exact
    ``backend="neuron"`` string path ``compile_blocks`` gave it.
    """
    try:
        fn = block._compiled_call_impl
        ctx = next(
            cell.cell_contents
            for cell in fn.__closure__
            if type(cell.cell_contents).__name__ == "OptimizeContext"
        )
        wrapper = ctx.callback._torchdynamo_orig_backend._torchdynamo_orig_backend
        return wrapper.compiler_name, wrapper.kwargs.get("options")
    except Exception as exc:  # noqa: BLE001 - reported as a FAIL by the caller
        return f"<unreadable: {exc!r}>", None


def check_broadcast(rank: int) -> bool:
    ok = True
    with track_fallbacks() as fallbacks:
        for dtype in BROADCAST_DTYPES:
            for shape in BROADCAST_SHAPES:
                expected = (torch.arange(math.prod(shape)) + 1).reshape(shape).to(dtype)
                got = broadcast_tensor(expected if rank == 0 else None, device=DEVICE).cpu()
                case_ok = (got.dtype == dtype and tuple(got.shape) == shape
                           and torch.equal(got, expected))
                report(rank, case_ok, f"broadcast {dtype} shape={shape}")
                ok &= case_ok
        edges = torch.tensor([INT32_MIN, INT32_MAX], dtype=torch.int64)
        got = broadcast_tensor(edges if rank == 0 else None, device=DEVICE).cpu()
        edges_ok = got.dtype == torch.int64 and torch.equal(got, edges)
        report(rank, edges_ok, f"broadcast torch.int64 int32-limit values arrive {got.tolist()}")
        ok &= edges_ok
        too_big = torch.tensor([1, 2**31], dtype=torch.int64)
        try:
            broadcast_tensor(too_big if rank == 0 else None, device=DEVICE)
            outcome = "completed"
        except RankFailureError:
            outcome = "rank_failure"
        except ValueError:
            outcome = "rejected"
        expected_outcome = "rejected" if rank == 0 else "rank_failure"
        range_ok = outcome == expected_outcome
        report(rank, range_ok, f"broadcast torch.int64 outside int32 range outcome={outcome}")
        ok &= range_ok
        # Still in step after the rejection: one more round trip must line up on every rank.
        after = torch.tensor([7.0, -7.0])
        got = broadcast_tensor(after if rank == 0 else None, device=DEVICE).cpu()
        step_ok = torch.equal(got, after)
        report(rank, step_ok, "broadcast in step after the rejection")
        ok &= step_ok
    report(rank, not fallbacks, f"broadcast fallbacks={fallbacks}")
    return ok and not fallbacks


def compiled_backend_ok(app) -> tuple[bool, str]:
    configs = [block_backend(block) for block in app.module.blocks]
    names = {name for name, _ in configs}
    options = [opts for _, opts in configs]
    shared = all(opts is options[0] for opts in options)
    first = options[0] if options else None
    ok = (
        names == {"neuron"}
        and shared
        and isinstance(first, dict)
        and first.get("fallback_execution") is False
        and "dynamic" in first  # written by the neuron backend: it received this dict
    )
    return ok, f"backend={sorted(names)} options={first} shared={shared}"


def check_lifecycle(rank, parallel, ckpt_dir, exec_mode, dtype, x, ref, ref_bf16) -> tuple:
    app = CountingToyApplication(
        model_path=ckpt_dir, parallel=parallel, dtype=dtype, exec_mode=exec_mode,
        device="neuron", n_blocks=N_BLOCKS, dim=DIM, hidden=HIDDEN, batch=BATCH, seq=SEQ,
    )
    with track_fallbacks() as fallbacks:
        start = time.perf_counter()
        app.load()
        load_s = time.perf_counter() - start
        xb = broadcast_tensor(x.to(dtype) if rank == 0 else None, device=DEVICE)
        out = app(xb).float().cpu()
        out_rank0 = broadcast_tensor(out if rank == 0 else None, device=DEVICE).cpu()
    same = torch.equal(out, out_rank0)
    numerics_ok, detail = numerics(out, ref, ref_bf16, dtype)
    compile_mode = exec_mode == "compile"
    blocks = [f"blocks.{i}" for i in range(N_BLOCKS)] if compile_mode else []
    lifecycle_ok = (
        app.is_loaded
        and app.compiled_blocks == blocks
        and app.warmup_shapes == [(BATCH, SEQ, DIM)]
        and all(p.device.type == "neuron" and p.dtype == dtype for p in app.module.parameters())
    )
    ok = numerics_ok and same and lifecycle_ok and not fallbacks
    extra = ""
    if compile_mode:
        backend_ok, backend_detail = compiled_backend_ok(app)
        warm, again = app.forward_neffs[0], app.forward_neffs[1]
        neff_ok = sum(warm) >= 1 and again == (0, 0)
        ok = ok and backend_ok and neff_ok and app.unwarmed_shapes == []
        extra = (f" {backend_detail} warmup_neffs(compiled,persistent)={warm} "
                 f"same_shape_neffs={again}")
    report(rank, ok, f"lifecycle mode={exec_mode} dtype={dtype} {detail} "
                     f"same_on_all_ranks={same} lifecycle_ok={lifecycle_ok} "
                     f"load_s={load_s:.1f} warmup_s={app.phase_seconds['warmup forward']:.1f} "
                     f"fallbacks={fallbacks}{extra}")
    return ok, app


def check_unwarmed_shape(rank, app, x2, ref2, ref2_bf16) -> bool:
    with track_fallbacks() as fallbacks:
        xb = broadcast_tensor(x2.to(app.dtype) if rank == 0 else None, device=DEVICE)
        out = app(xb).float().cpu()
    numerics_ok, detail = numerics(out, ref2, ref2_bf16, app.dtype)
    neffs = app.forward_neffs[-1]
    expected = [((BATCH, SEQ_UNWARMED, DIM),)]
    ok = app.unwarmed_shapes == expected and sum(neffs) >= 1 and numerics_ok and not fallbacks
    report(rank, ok, f"unwarmed shape mode={app.exec_mode} dtype={app.dtype} "
                     f"unwarmed_shapes={app.unwarmed_shapes} neffs(compiled,persistent)={neffs} "
                     f"{detail} fallbacks={fallbacks}")
    return ok


def numerics(out, ref, ref_bf16, dtype) -> tuple[bool, str]:
    if dtype == torch.float32:
        err = (out - ref).abs().max().item() / max(1.0, ref.abs().max().item())
        return err <= FP32_REL_TOL, f"rel_err={err:.1e} (tol {FP32_REL_TOL:.0e})"
    dev_err = (out - ref).abs().mean().item()
    cpu_err = (ref_bf16 - ref).abs().mean().item()
    ok = dev_err <= BF16_ERR_FACTOR * cpu_err + 1e-6
    return ok, f"mean_err device={dev_err:.2e} cpu_bf16={cpu_err:.2e}"


def check_failure(rank, parallel, ckpt_dir, exec_mode, inject) -> bool:
    app = ToyApplication(model_path=ckpt_dir, parallel=parallel, dtype=torch.bfloat16,
                         exec_mode=exec_mode, device="neuron", n_blocks=N_BLOCKS, dim=DIM,
                         hidden=HIDDEN, batch=BATCH, seq=SEQ, inject_failure=inject)
    start = time.perf_counter()
    try:
        app.load()
        outcome, message = "loaded", ""
    except RankFailureError as exc:
        outcome, message = "rank_failure", str(exc)
    except RuntimeError as exc:
        outcome, message = "raised", str(exc)
    elapsed = time.perf_counter() - start
    if inject == "forward":
        # Only the failing rank gets here; its peers stay in block 0's MLP all-reduce.
        failing = rank == FORWARD_FAILURE_RANK
        ok = (outcome == "raised" and f"injected failure: forward on rank {rank}" in message
              if failing else outcome != "loaded")
    elif rank == 0:
        ok = outcome == "raised" and f"injected failure: {inject}" in message
    else:
        ok = outcome == "rank_failure" and "failed on rank 0" in message
    ok = ok and not app.is_loaded
    print(f"{'PASS' if ok else 'FAIL'} inject={inject} rank={rank} outcome={outcome} "
          f"after {elapsed:.1f}s: {message}", flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exec-mode", choices=("eager", "compile"), default="eager")
    parser.add_argument("--inject-failure", choices=INJECT_FAILURES, default=None)
    parser.add_argument(
        "--work-dir", default=os.path.join(tempfile.gettempdir(), "difflet_c8_check")
    )
    args = parser.parse_args()
    if args.inject_failure is not None and args.exec_mode != "eager":
        parser.error("--inject-failure runs with --exec-mode eager")

    # Host-only TP1 references, built BEFORE the process group exists: once a 4-rank group is
    # up, an unsharded ToyBlocksModel cannot be built (its TP layers would size for tp=4).
    x = toy_blocks_input(BATCH, SEQ, DIM)
    weights, ref, ref_bf16 = toy_blocks_reference(N_BLOCKS, DIM, HIDDEN, x)
    x2 = toy_blocks_input(BATCH, SEQ_UNWARMED, DIM, seed=2)
    _, ref2, ref2_bf16 = toy_blocks_reference(N_BLOCKS, DIM, HIDDEN, x2)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    import torch_neuronx.metrics as metrics

    metrics.set_enabled(True)
    rank, _ = world_info()
    ckpt_dir = Path(args.work_dir) / "ckpt"

    def write_checkpoint() -> None:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        ckpt_dir.mkdir(parents=True)
        write_toy_checkpoint(ckpt_dir, weights, num_files=2)

    # Rank-0 host I/O; the phase's closing status all-reduce is the "checkpoint written" barrier.
    rank0_call(write_checkpoint, device=DEVICE, what="write checkpoint")

    summary = ""
    if args.inject_failure is not None:
        ok = check_failure(rank, parallel, ckpt_dir, args.exec_mode, args.inject_failure)
        if args.inject_failure == "forward":
            # Leave as a crashed rank would: no status collective, no process-group teardown
            # (its peers are inside an all-reduce). torchrun then stops them.
            print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} exiting with "
                  f"{FORWARD_FAILURE_EXIT if ok else 1}", flush=True)
            return FORWARD_FAILURE_EXIT if ok else 1
    else:
        ok = check_broadcast(rank)
        neffs = {}
        for dtype in (torch.float32, torch.bfloat16):
            passed, app = check_lifecycle(rank, parallel, ckpt_dir, args.exec_mode, dtype, x,
                                          ref, ref_bf16)
            ok &= passed
            neffs[str(dtype).removeprefix("torch.")] = app.forward_neffs
        if args.exec_mode == "compile":
            ok &= check_unwarmed_shape(rank, app, x2, ref2, ref2_bf16)
            # This rank's (compiled, persistent) per forward: warm-up, same shape[, new shape].
            summary = f" neffs={neffs}"
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'}{summary}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
