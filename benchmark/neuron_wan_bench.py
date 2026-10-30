"""Wan2.1 T2V on the neuron backend: one benchmark launch, one process per NeuronCore (torchrun).

Every rank brings the runtime up (``prepare_runtime``), builds ``TorchNeuronWanApplication``
in compile mode (40 blocks compiled in place into one graph, warm-up at the static shape),
then runs the real ``WanOrchestrator`` generate ``--iters`` times in the same process: the
first generate is this launch's number (cold or warm, decided by the NEFF cache the driver
hands it), the later ones are the resident rows (the serving steady state). Rank 0 decodes
every generate's latents on the host with the CLI's own ``_decode_latents_host`` (the
``--host-vae`` path) and writes ``iter<i>.mp4``; ranks 1-3 wait for that on a marker file,
not in a collective, so their next generate starts with rank 0's. Every rank writes
``result-rank<r>.json``; ``benchmark/neuron_cold_warm.py`` (the driver, which never touches
the device) reads them.

What each rank measures (all host wall clocks, ``time.perf_counter``):

* the compile/cache evidence: ``torch_neuronx.metrics`` enabled before ``prepare_runtime``,
  the ``CompilationCache.*`` counters at each snapshot (before/after ``prepare_runtime``,
  after ``load()``, around every generate, at the end), the ``CompilationTime`` samples, the
  runtime's per-NEFF cache entries, Dynamo's per-graph records and phase timers, the
  application's ``phase_seconds`` (``warmup forward`` holds the cold block compile);
* the DiT step: ``benchmark.harness.RealLoopStepTimer`` with ``torch.neuron.synchronize`` as
  its sync, stamped once per denoise iteration by the application's ``step_hook`` (after the
  orchestrator's scheduler step: at guidance > 1 one iteration is two DiT forwards plus the
  host UniPC step), the device synchronized once before the loop; inter-step deltas, step 0
  excluded (n = steps - 1). Each DiT forward is clocked as well (``forward_hook``);
* stage clocks: every ``encode_prompt`` (the prompt, and at guidance > 1 the empty negative
  prompt the orchestrator encodes lazily inside its first denoise iteration), rank 0's umT5
  compute, ``_denoise``, and rank 0's host decode (VAE decode, mp4 write);
* evidence per generate: ``track_fallbacks()``, ``unwarmed_shapes``, ``max_memory_allocated``,
  the sha256 of the final latents (every rank; never per step), and on rank 0 the sysfs
  throttle strings before the generate, right after it and after the decode.

The NEFF/HLO cache locations must already be in the environment when this process imports
torch (``TORCH_NEURONX_{NEFF_CACHE_DIR,NEFF_LOCAL_CACHE_DIR,HLO_CACHE_DIR}``, exported by the
driver from ``difflet.backends.neuron.compile.neuron_cache_env``): torch-neuronx fixes its HLO
lock directory at import. The script refuses to run without them.

Launch (the driver builds this; one device job at a time; absolute paths):

    HF_HUB_OFFLINE=1 DIFFLET_NO_JEMALLOC=1 OMP_NUM_THREADS=3 DIFFLET_NEURON_PG_TIMEOUT_S=3600 \\
    DIFFLET_COMPILE_CACHE=<cache> TORCH_NEURONX_NEFF_CACHE_DIR=<cache>/neuron/neff \\
    TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=<cache>/neuron/neff_local \\
    TORCH_NEURONX_HLO_CACHE_DIR=<cache>/neuron/hlo NKI_ENABLE_TRACE_CACHE=0 \\
    PYTHONSAFEPATH=1 PYTHONPATH=<worktree> <venv>/bin/python -m torch.distributed.run \\
        --standalone --nproc-per-node 4 <worktree>/benchmark/neuron_wan_bench.py \\
        --model-dir <snapshot> --steps 50 --guidance-scale 5.0 --iters 3 --out-dir <dir>

CPU helper (no torchrun, no device): ``--inspect-latents <latents.pt> [--inspect-mp4 <mp4>]``
prints one JSON line with the tensor's sha256 (the W4 rule: raw bytes, bf16 through int16),
dtype, shape and stats, the file's sha256, and the mp4 read back (frames, fps, codec, size).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_START = time.perf_counter()
SCRIPT_START_EPOCH = time.time()
os.environ["DIFFLET_BACKEND"] = "neuron"

import torch  # noqa: E402  (after DIFFLET_BACKEND; the cache variables come from the launcher)

MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
REVISION = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"
#: benchmark/models.py MATRIX["wan_2_1"].prompt
PROMPT = "a cinematic shot of a red fox running through a snowy forest"
CACHE_ENV_VARS = (
    "TORCH_NEURONX_NEFF_CACHE_DIR",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
    "TORCH_NEURONX_HLO_CACHE_DIR",
)
COUNTERS = ("TotalCompilations", "PersistentHits", "PersistentMisses", "InMemoryHits",
            "InMemoryMisses")
ENV_RECORDED = (
    "OMP_NUM_THREADS", "HF_HUB_OFFLINE", "DIFFLET_NO_JEMALLOC", "DIFFLET_NEURON_PG_TIMEOUT_S",
    "NEURON_RT_VISIBLE_CORES", "NEURON_RT_NUM_CORES", "DIFFLET_COMPILE_CACHE",
    *CACHE_ENV_VARS, "NKI_ENABLE_TRACE_CACHE", "NEURON_CC_FLAGS",
    "TORCH_NEURONX_ENABLED_METRIC_TABLES", "TORCH_NEURONX_METRICS_DIR", "DIFFLET_BACKEND",
    "DIFFLET_EXEC_MODE", "DIFFLET_DISABLE_PREWARM", "PYTHONPATH", "PYTHONSAFEPATH",
)
SYSFS_DEVICE = Path("/sys/class/neuron_device/neuron0")
GIB = 2**30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inspect-latents", type=Path, default=None,
                   help="CPU helper: describe a latents.pt and exit (no device)")
    p.add_argument("--inspect-mp4", type=Path, default=None,
                   help="CPU helper: with --inspect-latents, also read this mp4 back")
    p.add_argument("--model-dir", type=Path, help="the diffusers snapshot directory")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--text-seq-len", type=int, default=512)
    p.add_argument("--exec-mode", choices=("compile", "eager"), default="compile")
    p.add_argument("--cfg-parallel", action="store_true",
                   help="TP (world/2) x CFG 2: the guidance branches on separate rank groups")
    p.add_argument("--cp-degree", type=int, default=1,
                   help="context-parallel degree; TP is world / (cp * cfg)")
    p.add_argument("--cp-mode", choices=("ulysses", "gather_kv", "ring"), default="ulysses")
    p.add_argument("--sp", action="store_true", help="sequence parallelism within the TP group")
    p.add_argument("--iters", type=int, default=3,
                   help="generates in this process: the 1st is the launch's number, the "
                        "rest are the resident rows")
    p.add_argument("--encoder-threads", type=int, default=12,
                   help="rank 0's umT5 intra-op threads (pinned, D70: the embeddings' bits "
                        "depend on it)")
    p.add_argument("--decode-threads", type=int, default=12,
                   help="rank 0's torch threads for the host VAE decode")
    p.add_argument("--expect", choices=("cold", "warm"), default=None,
                   help="recorded only; the driver judges the counter rule")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--wait-timeout", type=float, default=1800.0,
                   help="seconds ranks 1-3 wait for rank 0's decode of one generate")
    args = p.parse_args(argv)
    if args.inspect_latents is None:
        if args.model_dir is None or args.out_dir is None:
            p.error("--model-dir and --out-dir are required (or --inspect-latents)")
        if args.iters < 1 or args.steps < 2:
            p.error("--iters must be >= 1 and --steps >= 2 (no step delta otherwise)")
    return args


# --------------------------------------------------------------------------- measurements


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Of the raw bytes; bf16 goes through an int16 view (numpy has no bf16). W4's rule."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    x = tensor.detach().to(torch.float32)
    return {"shape": list(x.shape), "dtype": str(tensor.dtype),
            "finite": bool(torch.isfinite(x).all()), "min": float(x.min()),
            "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std())}


def host_memory() -> dict[str, float]:
    """This process's VmHWM / VmRSS and the host's MemAvailable / Cached, in GiB."""
    values: dict[str, float] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmHWM:", "VmRSS:")):
            key, value = line.split(":", 1)
            values[f"{key}_gib"] = int(value.split()[0]) / 2**20
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:", "Cached:")):
            key, value = line.split(":", 1)
            values[f"host_{key}_gib"] = int(value.split()[0]) / 2**20
    return values


def counters() -> dict[str, int]:
    import torch_neuronx.metrics as metrics

    return {name: metrics.get_counter_value("CompilationCache." + name) or 0
            for name in COUNTERS}


def counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {name: after[name] - before[name] for name in after}


def dynamo_stats() -> dict[str, int]:
    from torch._dynamo.utils import counters as dynamo_counters

    return {"unique_graphs": int(dynamo_counters["stats"]["unique_graphs"]),
            "graph_breaks": int(sum(dynamo_counters["graph_break"].values()))}


def count_files(directory: str | None, pattern: str = "*") -> int | None:
    if not directory or not Path(directory).is_dir():
        return None
    return sum(1 for p in Path(directory).rglob(pattern) if p.is_file())


def device_memory() -> dict[str, float]:
    import torch_neuronx

    return {"allocated_gib": torch_neuronx.memory_allocated() / GIB,
            "max_allocated_gib": torch_neuronx.max_memory_allocated() / GIB}


def snapshot(t0: float, *, device: bool) -> dict[str, Any]:
    return {
        "t_s": time.perf_counter() - t0,
        "counters": counters(),
        "dynamo": dynamo_stats(),
        "neff_files": count_files(os.environ.get(CACHE_ENV_VARS[0]), "*.neff"),
        "hlo_files": count_files(os.environ.get(CACHE_ENV_VARS[2])),
        "device_memory": device_memory() if device else None,
        "host_memory": host_memory(),
    }


def sysfs_throttle() -> dict[str, str]:
    """Raw driver strings: every core's ham_throttle and the device power utilization.
    Their semantics are undocumented (research-benchmark 3.3): a changed ``entries``/``max``
    is a flag, not a verdict."""
    values: dict[str, str] = {}
    for core in sorted(SYSFS_DEVICE.glob("neuron_core*")):
        path = core / "stats/hardware/ham_throttle/throttle"
        try:
            values[core.name] = path.read_text().strip()
        except OSError as exc:
            values[core.name] = f"unreadable: {exc}"
    try:
        values["power_utilization"] = (SYSFS_DEVICE / "stats/hardware/power/utilization"
                                       ).read_text().strip()
    except OSError as exc:
        values["power_utilization"] = f"unreadable: {exc}"
    return values


def compile_info() -> dict[str, Any]:
    """The runtime's NEFF cache entries (``compilation_time_ms`` > 0: this process compiled
    it; 0: loaded from the persistent cache), the ``CompilationTime`` samples, Dynamo's
    per-graph records (``graph_id`` is the block NEFF's cache key) and Dynamo's phase timers.
    Concurrent compiles overlap: none of these sums is a wall clock."""
    import torch_neuronx
    import torch_neuronx.metrics as metrics
    from torch._dynamo.utils import compilation_time_metrics
    from torch_neuronx.neuron_dynamo_backend import metrics as dynamo_metrics

    info: dict[str, Any] = {}
    try:
        data = metrics.get_metric_data("CompilationCache.CompilationTime")
        info["compilation_time"] = None if data is None else {
            "accumulator_s": data["accumulator"] / 1e9,
            "total_samples": data["total_samples"],
            "samples_s": [s["value"] / 1e9 for s in data["samples"]],
        }
        info["cache_entries"] = [
            {"cache_key": e["cache_key"], "neff_size_bytes": int(e["neff_size_bytes"]),
             "access_count": int(e["access_count"]),
             "compilation_time_ms": int(e["compilation_time_ms"])}
            for e in torch_neuronx._C._get_compilation_cache_entries()]
        info["dynamo_graphs"] = [dataclasses.asdict(m)
                                 for m in dynamo_metrics.get_dynamo_metrics()]
        phases = {}
        for name, values in list(compilation_time_metrics.items()):
            values = [float(v) for v in values]
            phases[name] = {"total_s": sum(values), "count": len(values)}
        info["dynamo_phases_s"] = phases
    except Exception as exc:  # noqa: BLE001 - recorded; the driver reports a missing breakdown
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


# ------------------------------------------------------------------------------ decode


def decode_on_host(args: argparse.Namespace, latents_path: Path, mp4_path: Path
                   ) -> dict[str, Any]:
    """``_decode_latents_host`` (the CLI's --host-vae path: VAE load, decode, mp4 write) at
    ``--decode-threads``, timed, with the decoded frames described before the export."""
    from diffusers import AutoencoderKLWan

    from difflet.cli.orchestrators import wan as wan_cli

    timings: dict[str, float] = {}
    frames_info: dict[str, Any] = {}
    original_decode = AutoencoderKLWan.decode
    original_save = wan_cli._save_video

    def timed_decode(self, *decode_args, **decode_kwargs):
        begin = time.perf_counter()
        result = original_decode(self, *decode_args, **decode_kwargs)
        timings["vae_decode_s"] = time.perf_counter() - begin
        return result

    def described_save(tensor, output_path):
        frames = tensor.detach().to(torch.float32)
        frames_info.update(tensor_stats(frames))
        clipped = frames.clamp(-1, 1)
        frames_info["per_frame_std"] = [float(clipped[:, :, i].std())
                                        for i in range(frames.shape[2])]
        begin = time.perf_counter()
        saved = original_save(tensor, output_path)
        timings["mp4_write_s"] = time.perf_counter() - begin
        frames_info["saved_mp4"] = bool(saved)
        return saved

    previous = torch.get_num_threads()
    torch.set_num_threads(args.decode_threads)
    AutoencoderKLWan.decode = timed_decode
    wan_cli._save_video = described_save
    begin = time.perf_counter()
    try:
        wan_cli._decode_latents_host(str(latents_path), args.model_id, str(mp4_path),
                                     revision=args.revision)
    finally:
        AutoencoderKLWan.decode = original_decode
        wan_cli._save_video = original_save
        torch.set_num_threads(previous)
    timings["decode_latents_host_s"] = time.perf_counter() - begin
    exists = mp4_path.is_file()
    return {"timings": timings, "frames": frames_info, "threads": args.decode_threads,
            "mp4": str(mp4_path), "mp4_exists": exists,
            "mp4_bytes": mp4_path.stat().st_size if exists else None,
            "mp4_sha256": file_sha256(mp4_path) if exists else None}


def wait_for(marker: Path, timeout: float) -> float:
    """Ranks 1-3: sleep-poll for rank 0's marker file (no collective, no busy wait)."""
    begin = time.perf_counter()
    while not marker.exists():
        if time.perf_counter() - begin > timeout:
            raise TimeoutError(f"rank 0 did not write {marker} within {timeout:.0f}s")
        time.sleep(0.05)
    return time.perf_counter() - begin


# ----------------------------------------------------------------------------- the run


def run(args: argparse.Namespace) -> int:
    missing = [name for name in CACHE_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            f"{', '.join(missing)} not set: torch-neuronx fixes its HLO lock directory when "
            "this process imports torch, so the launcher must export the three cache "
            "locations (benchmark/neuron_cold_warm.py does, from neuron_cache_env)")
    import torch.distributed as dist
    import torch_neuronx
    import torch_neuronx.metrics as metrics

    from benchmark.harness import RealLoopStepTimer
    from difflet.backends.neuron.runtime import track_fallbacks
    from difflet.backends.registry import get_backend
    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.numerical import neuron_parity as parity

    import importlib

    # Fixed when torch-neuronx imported it: the HLO lock directory (R1 finding 1).
    hlo_lock_dir = importlib.import_module(
        "torch_neuronx.python_ops.compilation.hlo_cache").CACHE_DIR
    imports_s = time.perf_counter() - SCRIPT_START

    # Counters on before prepare_runtime (which may compile or load a NEFF itself), on a
    # runtime nothing has started yet.
    metrics.set_enabled(True)
    metrics_setup = {"enabled_before_prepare_runtime": bool(metrics.is_enabled()),
                     "runtime_initialized_when_enabled":
                         bool(torch_neuronx.is_neuron_runtime_initialized())}
    t0 = time.perf_counter()  # every *_s / t_s below is relative to prepare_runtime's start
    prepare_epoch = time.time()
    snaps: dict[str, Any] = {"pre_prepare": snapshot(t0, device=False)}
    world = int(os.environ.get("WORLD_SIZE", "1"))
    cfg_parallel = bool(getattr(args, "cfg_parallel", False))
    cp = int(getattr(args, "cp_degree", 1) or 1)
    lanes = cp * (2 if cfg_parallel else 1)
    if world % lanes:
        raise SystemExit(f"world size {world} is not divisible by cp * cfg = {lanes}")
    parallel = DiffletParallelConfig(tp_degree=world // lanes, cp_degree=cp,
                                     cp_mode=getattr(args, "cp_mode", "ulysses") if cp > 1 else "gather_kv",
                                     cfg_parallel_enabled=cfg_parallel,
                                     sp_enabled=bool(getattr(args, "sp", False)))
    get_backend("neuron").prepare_runtime(parallel)
    prepare_s = time.perf_counter() - t0
    rank = dist.get_rank() if dist.is_initialized() else 0
    snaps["after_prepare"] = snapshot(t0, device=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    default_threads = torch.get_num_threads()

    app = TorchNeuronWanApplication(
        model_path=args.model_dir, parallel=parallel, dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width, "num_frames": args.num_frames},
        text_seq_len=args.text_seq_len, exec_mode=args.exec_mode, device="neuron",
        encoder_threads=args.encoder_threads,
    )
    begin = time.perf_counter()
    with track_fallbacks() as load_fallbacks:
        app.load(skip_warmup=False)  # DiT: shards, per-block compile, warm-up; umT5 on rank 0
    torch.neuron.synchronize()
    load_s = time.perf_counter() - begin
    load_fallbacks = list(load_fallbacks)
    snaps["after_load"] = snapshot(t0, device=True)
    compile_after_load = compile_info()
    print(f"[bench rank {rank}] loaded in {load_s:.1f}s phases="
          f"{ {k: round(v, 1) for k, v in app.phase_seconds.items()} } counters="
          f"{counter_delta(snaps['after_load']['counters'], snaps['pre_prepare']['counters'])}"
          f" HBM max {snaps['after_load']['device_memory']['max_allocated_gib']:.2f} GiB",
          flush=True)

    # ------------------------------------------------------------- instrumentation
    pipeline, adapter, encoder = app.pipeline, app.transformer_adapter, app.text_encoder
    timer = RealLoopStepTimer(sync=torch.neuron.synchronize)
    app.step_hook = timer.step  # once per denoise iteration (after the scheduler step)
    forwards: list[float] = []
    adapter.forward_hook = forwards.append  # once per DiT forward

    encodes: list[dict[str, Any]] = []
    original_encode_prompt = pipeline.encode_prompt

    def timed_encode_prompt(*args, **kwargs):
        start = time.perf_counter()
        embeds = original_encode_prompt(*args, **kwargs)
        if args or kwargs.get("prompt") is not None or kwargs.get("input_ids") is not None:
            # the generate's first encode is its prompt; later ones are the negative prompt
            # (the orchestrator encodes it lazily inside the first denoise iteration)
            encodes.append({"tag": "negative" if encodes else "prompt",
                            "t_s": start - t0, "seconds": time.perf_counter() - start})
        return embeds

    pipeline.encode_prompt = timed_encode_prompt

    umt5: list[dict[str, Any]] = []  # rank 0: the encoder's own compute
    original_umt5 = encoder._encode

    def timed_umt5(input_ids, attention_mask):
        start = time.perf_counter()
        embeds = original_umt5(input_ids, attention_mask)
        umt5.append({"tokens": int(attention_mask.sum()), "threads": encoder.threads,
                     "seconds": time.perf_counter() - start})
        return embeds

    encoder._encode = timed_umt5

    denoise_marks: dict[str, float] = {}
    original_denoise = pipeline._denoise

    def timed_denoise(*args, **kwargs):
        torch.neuron.synchronize()  # no queued work leaks into the loop's first stamp
        denoise_marks["start"] = time.perf_counter()
        latents = original_denoise(*args, **kwargs)
        denoise_marks["end"] = time.perf_counter()
        return latents

    pipeline._denoise = timed_denoise

    # ------------------------------------------------------------- the generates
    iterations: list[dict[str, Any]] = []
    all_fallbacks: list[str] = []
    for index in range(1, args.iters + 1):
        timer.stamps.clear()
        forwards.clear()
        encodes.clear()
        umt5.clear()
        denoise_marks.clear()
        throttle_before = sysfs_throttle() if rank == 0 else None
        torch.neuron.synchronize()
        before = counters()
        start = time.perf_counter()
        with track_fallbacks() as fallbacks:
            out = app(
                generator=torch.Generator().manual_seed(args.seed),
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                output_type="latent",
            )
        end = time.perf_counter()
        throttle_after = sysfs_throttle() if rank == 0 else None  # before the idle decode
        fallbacks = list(fallbacks)
        all_fallbacks.extend(fallbacks)
        after = counters()
        latents = out.latents.detach().to(torch.float32)
        record: dict[str, Any] = {
            "index": index,
            "t_start_s": start - t0,
            "t_latents_s": end - t0,
            "generate_s": end - start,
            "encodes": list(encodes),
            "umt5_encodes": list(umt5),
            "denoise_s": denoise_marks["end"] - denoise_marks["start"],
            "iteration0_s": (timer.stamps[0] - denoise_marks["start"]) if timer.stamps else None,
            "stamps": len(timer.stamps),
            "step_deltas_s": timer.deltas(),
            "forwards_s": list(forwards),
            "counters": counter_delta(after, before),
            "fallbacks": fallbacks,
            "unwarmed_shapes": [[list(s) for s in key] for key in app.unwarmed_shapes],
            "max_memory_allocated_gib": device_memory()["max_allocated_gib"],
            "latents_sha256": tensor_sha256(latents),
            "latents_stats": tensor_stats(latents),
            "throttle_before": throttle_before,
            "throttle_after": throttle_after,
        }
        marker = args.out_dir / f"iter{index}.done"
        if rank == 0:
            latents_path = args.out_dir / f"iter{index}_latents.pt"
            torch.save(latents.cpu(), latents_path)
            record["latents_pt"] = str(latents_path)
            record["latents_pt_sha256"] = file_sha256(latents_path)
            mp4_path = args.out_dir / f"iter{index}.mp4"
            decode_start = time.perf_counter()
            record["decode"] = decode_on_host(args, latents_path, mp4_path)
            record["t_mp4_saved_s"] = time.perf_counter() - t0
            record["decode"]["wall_s"] = time.perf_counter() - decode_start
            record["throttle_after_decode"] = sysfs_throttle()
            marker.write_text(json.dumps({"epoch": time.time()}))
        else:
            record["waited_for_rank0_s"] = wait_for(marker, args.wait_timeout)
        record["host_memory"] = host_memory()
        iterations.append(record)
        it = record
        deltas = it["step_deltas_s"]
        median = sorted(deltas)[len(deltas) // 2] if deltas else math.nan
        print(f"[bench rank {rank}] generate {index}/{args.iters}: {it['generate_s']:.1f}s "
              f"denoise {it['denoise_s']:.1f}s step~{median * 1000:.1f}ms forwards "
              f"{len(it['forwards_s'])} counters {it['counters']} fallbacks {fallbacks} "
              f"unwarmed {it['unwarmed_shapes']} latents {it['latents_sha256'][:16]}"
              + (f" decode {it['decode']['timings'].get('decode_latents_host_s', math.nan):.1f}s"
                 f" mp4 {it['decode']['mp4_sha256'][:16] if it['decode']['mp4_sha256'] else None}"
                 if rank == 0 else f" waited {it['waited_for_rank0_s']:.1f}s"), flush=True)

    snaps["end"] = snapshot(t0, device=True)
    evidence = parity.device_evidence(app, all_fallbacks)
    dit = app.transformer
    result: dict[str, Any] = {
        "rank": rank,
        "world_size": world,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "env": {name: os.environ.get(name) for name in ENV_RECORDED},
        "hlo_lock_dir_at_import": hlo_lock_dir,
        "torch_threads": {"default": default_threads, "encoder": args.encoder_threads,
                          "decode": args.decode_threads if rank == 0 else None},
        "epoch": {"script_start": SCRIPT_START_EPOCH, "prepare_start": prepare_epoch},
        "imports_s": imports_s,
        "prepare_runtime_s": prepare_s,
        "load_s": load_s,
        "metrics_setup": metrics_setup,
        "phase_seconds": dict(app.phase_seconds),
        "load_fallbacks": load_fallbacks,
        "compiled_blocks": len(app.compiled_blocks),
        "warmup_shapes": [list(s) for s in (dit.warmup_shapes or [])],
        "load_report": {k: len(v) for k, v in (dit.load_report or {}).items()},
        "snapshots": snaps,
        "counters": {
            "prepare_runtime": counter_delta(snaps["after_prepare"]["counters"],
                                             snaps["pre_prepare"]["counters"]),
            "load": counter_delta(snaps["after_load"]["counters"],
                                  snaps["after_prepare"]["counters"]),
            "run": counter_delta(snaps["end"]["counters"], snaps["pre_prepare"]["counters"]),
        },
        "compile_after_load": compile_after_load,
        "compile_end": compile_info(),
        "iterations": iterations,
        "evidence": evidence,
        "peak_device_mem_gib": snaps["end"]["device_memory"]["max_allocated_gib"],
        "t_end_s": time.perf_counter() - t0,
    }
    path = args.out_dir / f"result-rank{rank}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[bench rank {rank}] wrote {path}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


# --------------------------------------------------------------------------- CPU helper


def inspect(latents_path: Path, mp4_path: Path | None) -> int:
    """One JSON line: the latents tensor and file hashes, and the mp4 read back."""
    tensor = torch.load(latents_path, map_location="cpu", weights_only=True)
    info: dict[str, Any] = {"latents": str(latents_path),
                            "tensor_sha256": tensor_sha256(tensor),
                            "file_sha256": file_sha256(latents_path),
                            **tensor_stats(tensor)}
    if mp4_path is not None:
        import imageio.v2 as imageio
        import numpy as np

        reader = imageio.get_reader(str(mp4_path), format="ffmpeg")
        try:
            meta = reader.get_meta_data()
            frames = [np.asarray(frame) for frame in reader]
        finally:
            reader.close()
        info["mp4"] = {"path": str(mp4_path), "frames": len(frames), "fps": meta.get("fps"),
                       "codec": meta.get("codec"), "size": list(meta.get("size") or []),
                       "sha256": file_sha256(mp4_path), "bytes": mp4_path.stat().st_size}
    print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.inspect_latents is not None:
        sys.exit(inspect(parsed.inspect_latents, parsed.inspect_mp4))
    sys.exit(run(parsed))
