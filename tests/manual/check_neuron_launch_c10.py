"""Manual check (C10): the torchrun launch path runs the toy application on 4 ranks.

Run from the worktree root on the trn2 host, with no other process holding the cores
(``neuron-ls`` shows no process), ``WORK`` a scratch directory (default: a fresh temp dir):

    DIFFLET_BACKEND=neuron DIFFLET_DISABLE_PREWARM=1 PYTHONPATH=<worktree> timeout 3600 \
        /home/ubuntu/workspace/native_venv/bin/python tests/manual/check_neuron_launch_c10.py \
        --work-dir WORK

This driver is a single process and never touches the device. It writes the toy
checkpoint and an fp32 CPU reference, then for each exec mode calls
difflet.cli.runner.run_stage. DIFFLET_BACKEND=neuron selects its MPMD branch, which runs
`torchrun --standalone --nproc-per-node 4 -m difflet.cli.stage --orchestrator
tests.unit.backends._neuron_toy:ToyOrchestrator --stage toy --exec-mode <mode>`.
Every rank checks that the Neuron runtime is not up, goes through
DiffletPipeline.from_pretrained (non-AoT branch, per-rank core binding, neuron process group)
and the TorchNeuronApplicationBase lifecycle, runs one forward under track_fallbacks, prints
`[rank r] PASS|FAIL`, and exits non-zero on failure; rank 0 writes the output.

The ranks' compile cache is ``--compile-cache`` (default ``WORK/compile_cache``), passed as
DIFFLET_COMPILE_CACHE with TORCH_NEURONX_NEFF_CACHE_DIR unset, so prepare_runtime places the
NEFF cache at ``<compile-cache>/neuron/neff``. On a cold cache the compile launch must add
``.neff`` files there; no launch may add any to ``/tmp/neff_cache`` (torch-neuronx's default).

``--fail-rank R`` instead runs one eager launch in which rank R raises inside block 0 of the
warm-up forward (``DIFFLET_TOY_FAIL=forward:R``), between its two all-reduces: its stage
process must exit non-zero and torchrun must stop the other ranks, which are blocked in the
second all-reduce, and exit non-zero itself within ``--fail-within`` seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REL_TOL = 3e-2  # bf16 device vs fp32 CPU reference, relative to max |reference|
MIN_COSINE = 0.999
TORCH_NEURONX_DEFAULT_NEFF_CACHE = Path("/tmp/neff_cache")


def _compare(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    out64, ref64 = out.double().flatten(), ref.double().flatten()
    rel = ((out64 - ref64).abs().max() / ref64.abs().max().clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(out64, ref64, dim=0).item()
    return rel, cos


def count_neffs(directory: Path) -> int:
    return sum(1 for _ in directory.rglob("*.neff")) if directory.is_dir() else 0


def _launch(run_stage, toy, mode: str, work: Path, num_cores: int) -> None:
    run_stage(
        toy.TOY_ORCHESTRATOR,
        toy.TOY_STAGE,
        num_cores=num_cores,
        virtual_core_size=None,
        cli_args=["--exec-mode", mode, "--work-dir", str(work)],
    )


def _check_failure(run_stage, toy, work: Path, args) -> bool:
    os.environ[toy.TOY_FAIL_ENV] = f"forward:{args.fail_rank}"
    start = time.monotonic()
    try:
        _launch(run_stage, toy, "eager", work, args.num_cores)
    except subprocess.CalledProcessError as exc:
        returncode = exc.returncode
    else:
        returncode = 0
    finally:
        os.environ.pop(toy.TOY_FAIL_ENV, None)
    elapsed = time.monotonic() - start
    ok = returncode != 0 and elapsed <= args.fail_within and not (work / "result-eager.json").exists()
    print(
        f"{'PASS' if ok else 'FAIL'} launch-failure fail_rank={args.fail_rank} "
        f"torchrun_exit={returncode} elapsed={elapsed:.1f}s (limit {args.fail_within:.0f}s)",
        flush=True,
    )
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--exec-mode", dest="exec_modes", action="append", choices=["eager", "compile"]
    )
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--compile-cache", default=None)
    parser.add_argument("--fail-rank", type=int, default=None)
    parser.add_argument("--fail-within", type=float, default=120.0)
    args = parser.parse_args(argv)
    modes = args.exec_modes or ["eager", "compile"]

    if os.environ.get("DIFFLET_BACKEND") != "neuron":
        print("FAIL c10 launch: set DIFFLET_BACKEND=neuron (the torchrun branch is opt-in)")
        return 1
    os.environ.setdefault("DIFFLET_DISABLE_PREWARM", "1")

    from difflet.backends.neuron.compile import (
        NEFF_CACHE_ENV,
        NEFF_LOCAL_CACHE_ENV,
        default_neff_cache_dir,
    )
    from difflet.backends.neuron.runtime import _neuron_runtime_initialized
    from difflet.cli.runner import run_stage
    from tests.unit.backends import _neuron_toy as toy

    os.environ.pop(toy.TOY_DEVICE_ENV, None)  # the ranks must run on the neuron device
    os.environ.pop(toy.TOY_FAIL_ENV, None)
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="difflet-c10-")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cache = Path(args.compile_cache or work / "compile_cache").resolve()
    os.environ["DIFFLET_COMPILE_CACHE"] = str(cache)
    for name in (NEFF_CACHE_ENV, NEFF_LOCAL_CACHE_ENV):
        if os.environ.pop(name, None) is not None:
            print(f"note: unset {name} so the ranks derive it from DIFFLET_COMPILE_CACHE")
    neff_dir = default_neff_cache_dir()
    toy.prepare_toy_work_dir(work)
    reference = torch.load(work / "reference.pt")["output"]
    driver_runtime = _neuron_runtime_initialized()
    print(f"driver: runtime_initialized={driver_runtime} work_dir={work} neff_dir={neff_dir}",
          flush=True)
    if driver_runtime:
        print("FAIL c10 launch: the driver started the Neuron runtime")
        return 1

    if args.fail_rank is not None:
        return 0 if _check_failure(run_stage, toy, work, args) else 1

    cold = count_neffs(neff_dir) == 0
    tmp_before = count_neffs(TORCH_NEURONX_DEFAULT_NEFF_CACHE)
    ok = True
    outputs: dict[str, torch.Tensor] = {}
    for mode in modes:
        neffs_before = count_neffs(neff_dir)
        tmp_mode_before = count_neffs(TORCH_NEURONX_DEFAULT_NEFF_CACHE)
        start = time.monotonic()
        try:
            _launch(run_stage, toy, mode, work, args.num_cores)
        except subprocess.CalledProcessError as exc:
            print(f"FAIL launch-{mode} torchrun exited with {exc.returncode}", flush=True)
            ok = False
            continue
        elapsed = time.monotonic() - start
        new_neffs = count_neffs(neff_dir) - neffs_before
        tmp_new = count_neffs(TORCH_NEURONX_DEFAULT_NEFF_CACHE) - tmp_mode_before
        result = json.loads((work / f"result-{mode}.json").read_text())
        out = torch.load(work / result["output"])
        rel, cos = _compare(out, reference)
        want_blocks = [f"blocks.{i}" for i in range(toy.TOY_N_BLOCKS)] if mode == "compile" else []
        neff_ok = tmp_new == 0 and (new_neffs >= 1 if mode == "compile" and cold else True)
        case_ok = (
            result["exec_mode"] == mode
            and result["world_size"] == args.num_cores
            and result["backend"] == "neuron"
            and result["device"] == "neuron"
            and result["forward_ran"]
            and not result["fallbacks"]
            and not result["manifest_written"]
            and not result["runtime_initialized_before_load"]
            and result["compiled_blocks"] == want_blocks
            and result["warmup_shapes"] == [[1, toy.TOY_TOKENS, toy.TOY_DIM]]
            and neff_ok
            and rel <= REL_TOL
            and cos >= MIN_COSINE
        )
        print(
            f"{'PASS' if case_ok else 'FAIL'} launch-{mode} world={result['world_size']} "
            f"exec_mode={result['exec_mode']} backend={result['backend']} "
            f"compiled_blocks={len(result['compiled_blocks'])} fallbacks={result['fallbacks']} "
            f"manifest={result['manifest_written']} "
            f"runtime_up_before_load={result['runtime_initialized_before_load']} "
            f"new_neffs={new_neffs} (cache {'cold' if cold else 'warm'}) tmp_neff_cache_new={tmp_new} "
            f"rel_err={rel:.3e} cos={cos:.6f} elapsed={elapsed:.0f}s",
            flush=True,
        )
        ok = ok and case_ok
        outputs[mode] = out
    if len(outputs) == 2:
        rel, cos = _compare(outputs["compile"], outputs["eager"])
        pair_ok = rel <= REL_TOL and cos >= MIN_COSINE
        print(
            f"{'PASS' if pair_ok else 'FAIL'} launch-eager-vs-compile "
            f"rel_err={rel:.3e} cos={cos:.6f}",
            flush=True,
        )
        ok = ok and pair_ok
    total = count_neffs(neff_dir)
    tmp_total_new = count_neffs(TORCH_NEURONX_DEFAULT_NEFF_CACHE) - tmp_before
    cache_ok = total >= 1 and tmp_total_new == 0
    ok = ok and cache_ok
    print(
        f"{'PASS' if cache_ok else 'FAIL'} neff-cache neff_dir={neff_dir} neffs={total} "
        f"tmp_neff_cache_new={tmp_total_new}",
        flush=True,
    )
    print(
        f"{'PASS' if ok else 'FAIL'} c10 launch modes={','.join(modes)} "
        f"world={args.num_cores} work_dir={work}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
